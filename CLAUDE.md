# CLAUDE.md — gzippy Development Guide

## ⛔ ANTI-BIAS PREAMBLE — READ FIRST (the governing law; it overrides everything below it)

You are fine-tuned to produce confident conclusions from reasoning and to narrate
them as findings. **In THIS project that default has failed catastrophically — in a
single session it produced ~11 confident "the lever is X" conclusions, each disproven
by the very next measurement** (shared-floor, key-pin, prefetch-depth, "T1 settled",
single-shot-routing, async-overlap, decode-volume, utilization, B-width,
templated-Block, "fix clean-path overhead"). Every one felt rigorous. Every one was
wrong. The reasoning is not the problem to fix — the *trusting of reasoning* is. So:

**THE GOVERNING POLICY (user, 2026-06-14) — enforced, not advisory:**

1. **ONLY DETERMINISTIC SOFTWARE IS TRUSTED.** No hypothesis, theory, attribution,
   model, or conclusion may live in conversation or memory AS A FINDING. A finding
   EXISTS only as a FULCRUM MEASUREMENT that ran and passed its own self-tests. Prose
   is never where a conclusion lives.
2. **A claim of a lever / cause / win / tie / regression / "settled" / "dead" is VALID
   only if it is the OUTPUT of a Fulcrum measurement whose self-tests passed AND whose
   significance gate is met (Δ vs inter-run spread, N≥7, replicated on BOTH arches).**
   Anything else is an **OPEN HYPOTHESIS** — you MUST label it `HYPOTHESIS
   (unvalidated)`, and its ONLY permitted next action is to build/run the Fulcrum
   measurement that would test it. You may not act on it, bank it, ship from it, or
   stack another inference on top of it.
3. **FULCRUM IS THE SOLE ORACLE** — used soberly and humbly. Not advisors, not your
   reasoning, not a hand-rolled script, not a "code-verified" read (even reading the
   source is an inference that has been wrong — the shared-ness gate's empirical arm,
   not the cfg-read, is the verdict).
4. **THE WORK IS BUILDING THE TOOL THAT PERFORMS THE ANALYSIS — not performing the
   analysis yourself.** When you need an answer, the deliverable is the Fulcrum
   measurement (carrying its own tests) that answers it deterministically. "Fulcrum
   does our work." If Fulcrum cannot yet answer the question, EXTEND FULCRUM FIRST;
   do not substitute hand-computation, eyeballing, or intuition.

**Tripwire.** If you catch yourself writing "so the cause is…", "this means…",
"therefore the lever is…", "X is settled/dead", "this is shared/native-heavy",
"Δ≈0 so it's a win/tie" — STOP. That sentence is the bias firing. Rewrite it as a
`HYPOTHESIS (unvalidated)` line plus the exact Fulcrum measurement that would confirm
or falsify it. A correct (byte-identical) code change may be KEPT on a TIE, but "it
TIE'd" is NOT a finding about the cause.

Named biases this preamble exists to kill (all observed here): inference-as-conclusion
(the core one); premature closure / synthesis-ahead-of-evidence; over-correction
(flipping to the opposite conclusion just as confidently); attribution-as-verdict
(decompose/blame instead of causally perturb); dimensional/statistical error
(multiplying a CPU-share by a wall-time; calling Δ<spread a win; no significance test);
single-arch/single-corpus over-generalization (AMD-silesia → universal); trusting
un-self-validated instruments (file-sink A/B, inert oracle, hardcoded `pred_available`,
contaminated perf-annotate); over-banking interpretation (memory accreting disproven
"the lever is X" claims that misguide the next session).

## Prime Directive

**gzippy aims to be the fastest gzip implementation ever created — but the OPERATING
prime directive is: BUILD THE TOOL THAT MEASURES, don't hand-derive conclusions.**
Speed is the goal; a passing, self-validated Fulcrum measurement is the only currency
that counts toward it. Every session's primary work product is either (a) a
byte-identical code change whose effect a Fulcrum measurement confirmed at the wall,
or (b) an improvement to Fulcrum/the test harness that makes the next causal question
deterministically answerable. Prose analysis is not a work product.

## Goal (updated 2026-05-30 — perf-driven; faithful structural CONVERGENCE to rapidgzip REINSTATED)

**gzippy must be at least at parity with EVERY gzip/DEFLATE tool — gzip, pigz,
libdeflate, ISA-L, zlib-ng, AND rapidgzip — across archive × thread-count, with a
pure-Rust DEFLATE engine as the SOLE production decode path and C-FFI removed from
the decode graph (kept behind a dev/oracle feature as fuzz oracles).** Scope is the
GNU-gzip-family formats: gzip single-member, multi-member, BGZF, raw DEFLATE (inner
codec). BZIP2 and the ZLIB *stream decoder* are OUT (a harmless ZLIB header parser
may stay).

**Faithful structural convergence to rapidgzip is REINSTATED (2026-05-30) — the
2026-05-29 rescind is REVERSED.** That rescind rested entirely on the frozen
clean-window oracle ("gzippy's pipeline is ALREADY at rapidgzip parity, isal
clean-window 2035 ≈ rapidgzip 2067") — the instrument later proven BROKEN: it
silently re-ran the full bootstrap (fixed `64eb6df`). With that premise dead, so is
the decompose-a-slice-and-shave-it loop it licensed — that loop is how multiple
sessions were spent measuring TIEs. rapidgzip's source (`vendor/rapidgzip/`) is the
WORKING BLUEPRINT — faithfully port its structure.

rapidgzip carries the SAME u16 marker machinery gzippy does (measured on a traced
rapidgzip: 31.25% replaced-marker symbols, a 0.113s apply-window pass), and gzippy's
clean-window arming is a byte-for-byte port of vendor (`deflate.hpp:1282-1284` ↔
`deflate_block.rs:781-783`). So the gap is NOT extra machinery to delete — deleting
`deflate_block`/marker rings/`apply_window` would diverge from rapidgzip, not converge.

## Measurement PROTOCOL (ENFORCEABLE — a measurement that skips a gate is not a measurement)

This is no longer advice you weigh; it is the definition of "finding." A statement
about cause/lever/win/tie that did not pass EVERY gate below is an OPEN HYPOTHESIS
(label it as such) — not a result, not bankable, not actionable.

**GATE 0 — INSTRUMENT SELF-VALIDATION (BLOCKING; the single most-violated rule).**
Before any number is reported, the instrument must LOUD-PASS its own tests, else the
number does not exist:
  (a) every env knob/flag the run sets has a CONFIRMED consumer in `src/` (grep proves
      it — `GZIPPY_TIMELINE` vs `GZIPPY_LOG_FILE`, `GZIPPY_FORCE_PARALLEL_SM`, etc.
      have all been set inertly);
  (b) the comparator binary EXISTS on that box and self-tests to 1.0 ± spread
      (binary-vs-itself; "no rg on solvency" went unnoticed for a whole session);
  (c) any "oracle"/perturbation run produces output that PROVABLY DIFFERS from the
      baseline (counter fired, hits>0, hits==expected-count) — else it is inert and is
      silently measuring the normal path (`GZIPPY_SEED_WINDOWS` no-op-to-None);
  (d) both A/B arms use the SAME sink (`/dev/null`, matching how rg is measured) —
      a file sink penalizes the FASTER arm and manufactures phantom sign-flips;
  (e) conservation holds (busy+idle==span; buckets reconcile to chunk count; no
      double-count — the "62ms serial CRC" phantom was a nested-span double-count).
A self-validated instrument's ATTRIBUTION is STILL only attribution — it is a
HYPOTHESIS until Gate 2 (a tool proving it is internally consistent does NOT prove its
blame is the cross-tool causal lever).

**GATE 1 — STATISTICAL SIGNIFICANCE (BLOCKING).** Interleaved best-of-N≥7 on a frozen
host; report Δ AND inter-run spread. **Δ < spread ⇒ TIE, full stop — never a win.**
Never multiply a CPU-share by a wall-time, or a ratio by a ratio, to synthesize an
absolute (the decode-volume "1.33× more bytes" was circular share×wall arithmetic);
if you need an absolute, MEASURE the absolute.

**GATE 2 — CAUSAL PERTURBATION IS THE ONLY VERDICT (attribution never is).** To prove
region R gates the wall, change R's time by a KNOWN factor (≥2 magnitudes) and measure
the interleaved wall response: monotonic+proportional ⇒ on the critical path; flat ⇒
slack. ALWAYS run a frequency-neutral control (sleep that yields the core, not a busy
spin that depresses turbo); if the delta survives the sleep, it is real. Slow-down
slope ≠ speed-up ceiling — to BOUND a speed-up you must REMOVE the region (oracle) and
measure, never extrapolate a slope through an unlocated knee.

**GATE 3 — CROSS-ARCH / CROSS-CORPUS REPLICATION (BLOCKING for any banked finding).**
A one-arch result is NOT a finding ("deficit grows with T" held only on noisy Intel;
AMD showed native scaled like rg). A single-corpus or model-anchored result is NOT a
verdict (model is a hard-to-compress CORNER case). Replicate on the other arch and a
balanced corpus spread before the claim is anything but a HYPOTHESIS.

**GATE 4 — PRODUCTION-PATH ASSERTION (mechanical, every run).** `GZIPPY_DEBUG=1` →
`path=ParallelSM`; build `--no-default-features --features pure-rust-inflate`;
`GZIPPY_FORCE_PARALLEL_SM=1` to exercise the engine at every T; verify the binary's
feature-set fingerprint (a mislabeled native-as-isal binary produced a false "ISA-L
dormant" bombshell); sha-verified output (a speed win with wrong bytes is a loss).

**GATE 5 — EVIDENCE TIER + SCOPE STAMP (how you must price every claim).** Tier the
evidence and never over-bet: removal-oracle / causal-perturbation = STRONG;
cross-tool frozen matrix = STRONG; self-validated tool attribution = HYPOTHESIS;
whole-program perf attribution = WEAK; source-read = HYPOTHESIS. Every result is a
TIME-STAMPED verdict IN A CONTEXT (commit/bin-sha, corpus, arch, T, code state) — it
holds ONLY there. A disproof is not an eternal law: `git diff <cell-src-sha>..HEAD`
before relying on it (the 247ms "DIS-15" tax was real then, GONE at HEAD; citing it as
current sent the campaign down a wrong front).

**On REJECTING a direction.** "It TIE'd on this run" is a TIE verdict on THAT cell,
not a refutation of the direction. To DECLARE a direction dead you still need a
Gate-2 mechanism (a measured way it makes things worse, or a confirmed structural
reason it cannot help) — but equally, do NOT keep re-banking a "dead lever" as prose;
a dead direction is recorded as a FALSIFY entry with its premise + scope + re-open
trigger, never as a standing conclusion.

**Fulcrum is the SOLE oracle, and BUILDING it is the work.** `fulcrum vs`, `fulcrum
flow`, `fulcrum critpath`, `fulcrum causal`, `fulcrum score`/matrix, `fulcrum decide`,
`fulcrum locate` produce the numbers; each must carry its own self-tests (Gate 0) and
significance accounting (Gate 1). They are still HYPOTHESIS GENERATORS until Gate 2.
When a question can't be answered deterministically by Fulcrum, the correct next action
is to EXTEND FULCRUM (tooling is PULLED by a blocked finding, never pushed for its own
sake) — never to hand-roll a script or eyeball it (hand scripts manufacture phantoms).
The `plans/` directory was deleted as stale-prone interpretation; durable operational
facts live in `scripts/bench/guest.env` and git history. Treat EVERY "the lever is X"
claim — including ones written in memory — as an unvalidated hypothesis until a fresh
gated measurement confirms it THIS session.

Done when an Opus advisor agrees gzippy is at >=parity with every tool above on the
closable cells AND the pure-Rust decoder is the sole decode path with C-FFI off the
decode graph. **NEW GOAL (2026-05-30, user-set, supersedes the above hedges): FULLY PORT RAPIDGZIP
TO EXACT PERFORMANCE PARITY on the same workloads.** rapidgzip's source
(`vendor/rapidgzip/`) is the blueprint; a faithful structural port of its decode
pipeline is the method AND the done-criterion — done when gzippy matches rapidgzip's
wall (TIE-or-better, `scripts/measure.sh`, interleaved + sha-verified) across the
workload matrix (silesia-large × T1–T16, etc.), the structure faithfully mirrors
rapidgzip, and the pure-Rust decoder is the sole decode path.**

## Permission to fully reimplement the inner inflate

> **Confirm the wall moves with a causal perturbation BEFORE a work-stretch here**
> (Measurement PROCESS above). Do not pre-judge the inner loop as either the lever
> or a dead end from cached A/Bs — a +50–200% slow-injection of the window-absent
> bootstrap moves the wall ~proportionally (survived a frequency-neutral disproof),
> so the decode is on the critical path; whether SPEEDING it pays is bounded by the
> bootstrap-removed oracle, which must be repaired and run to set the ceiling.

The "port faithfully, don't innovate" rule is **scoped to architecture
and high-level shape** (chunk pipeline, prefetcher, block finder, etc.).
For the inner Huffman decode loop — `decode_huffman_body_resumable` and
the `LitLenTable` / `DistTable` / `Bits` primitives — full
re-implementation of every libdeflate / ISA-L technique is in scope and
explicitly authorized, including:

- Multi-literal lookahead (2-/3-/4-literal packed-write paths).
- Fixed-Huffman static-table specialization.
- BMI2 PEXT / BZHI runtime dispatch.
- Table prefetch (`_mm_prefetch`) ahead of dependent loads.
- Inline-asm hot loops if needed to match vendor codegen.
- Reorganized state-machine that elides the resumable yield-check tax
  when output has FASTLOOP_OUTPUT_MARGIN bytes of headroom.

Prior falsifications (e.g. commit `ca52389` SIMD multi-literal regression)
are **NOT binding** — they were measured against the pre-PRELOAD,
pre-BMI2 hot loop. Re-attempt any of them with fresh measurement.

### Update 2026-05-27 (later same day): innovation allowed in the inner loop

The "every change must have a vendor counterpart" requirement is
RESCINDED for the inner Huffman decode loop. Novel techniques that
have no vendor counterpart are now in scope provided they:

- Preserve correctness (all 635+ lib tests + corpus differential pass).
- Show measured win on the bench harness (no negative-results-allowed).
- Document the deviation in the commit message so the falsification
  record stays honest.

This still doesn't extend to architecture (chunk pipeline /
prefetcher / block finder / consumer) — those keep the vendor-port
rule. Only the inner Huffman loop and supporting primitives
(`LitLenTable`, `DistTable`, `Bits`, `bmi2.rs`) are open territory.

### Update 2026-05-27 (final): build the fastest possible raw Huffman decoder

Sharpening: the GOAL is *"build a provably-correct, fastest-possible
raw Huffman decoder"* (user directive). Implications:

- **Provably correct.** Strong testing — corpus differential against
  multiple independent oracles (flate2 + libdeflate), property-based
  testing with proptest, fuzzing if needed, plus the existing 635 lib
  tests. The decoder must produce byte-for-byte output identical to
  vendor on every legal input.
- **Fastest possible.** Beat ISA-L, libdeflate, AND flate2/zlib-ng on
  representative workloads. Not just "vendor-competitive" — strictly
  faster.
- **Arch-specific compilation is in scope.** Use `RUSTFLAGS="-C
  target-cpu=native"` for benches; build per-arch variants (BMI2 on
  x86_64, NEON on aarch64); per-CPU dispatch where it pays. Portable
  binaries can ship later via runtime dispatch; for the perf claim,
  arch-specific is the target.

The decoder is now decoupled (in principle) from the rest of the
parallel-SM machinery — it lives in `src/decompress/inflate/` and can
be evaluated as a standalone primitive. The resumable contract still
applies because callers in `parallel/` need it, but the contract's
cost should be amortized into the FASTLOOP and irrelevant to the
fastest-possible claim.

## Rules

1. **ONE PRODUCTION PATH** — know exactly which function the CLI calls. Test that function.
2. **RUN `make` FIRST** — before `make ship`, before committing. `make` catches regressions in 30s.
3. **BENCHMARK EVERYTHING** — `make ship` (homelab bench on `neurotic`) is authoritative; local `make` is for iteration.
4. **NEVER COMPROMISE CORRECTNESS** — output bytes, CRC32, ISIZE must always verify.
5. **NO FALLBACKS** — failure is an explicit `Err(GzippyError::Decompression(_))`. No silent libdeflate or ISA-L retries from the SM body. `decompress_single_member` either succeeds via the parallel pipeline or returns an error.
6. **FAITHFULLY PORT RAPIDGZIP TO EXACT PERFORMANCE PARITY (2026-05-30, user-set goal — reverses the prior "speed-not-fidelity").** rapidgzip's architecture is the fastest known and its source is in `vendor/rapidgzip/`: it is the BLUEPRINT to transliterate, not merely a reference to consult. Port its decode pipeline faithfully — consumer/chunk-lifecycle, block finder, window map, marker resolution, chunk decode — until gzippy's wall matches rapidgzip's on the same workloads. The prior "diverge freely / throughput not vendor parity" stance rested on the broken clean-window oracle (`64eb6df`) and is REVERSED. Keep gzippy's own inflate primitives ONLY where they meet-or-beat rapidgzip WITHOUT breaking the faithful structure; otherwise mirror the vendor `file:line`. Verify every change on `scripts/measure.sh` (interleaved, sha-verified) vs rapidgzip — never an internal slice; that is the rule the decompose-loop kept breaking.

## Production Routing (Apr 2026)

### Decompression

```
Input → decompress::mod: decompress_gzip_libdeflate
  ├─ gzippy-parallel? ("GZ" subfield in FEXTRA)
  │     → bgzf::decompress_bgzf_parallel (libdeflate FFI, T1 or Tmax internally)
  ├─ Multi-member? (trailing gzip headers detected)
  │     T1  → decompress_multi_member_sequential (libdeflate, member-by-member)
  │     Tmax → bgzf::decompress_multi_member_parallel (libdeflate FFI)
  └─ Single-member? (CORRECTED 2026-06-12 — the old "ISA-L + T>1 + >10 MiB"
        line was STALE: MIN_PARALLEL_COMPRESSED is dead code, there is NO size
        gate at HEAD, and a field-matrix worker re-derived the phantom
        threshold from this table. Current truth, mod.rs:185-232:)
        parallel_sm build (features pure-rust-inflate / gzippy-isal):
            EVERY single-member at EVERY T → ParallelSM
            (gzippy-isal at T1 → IsalSingleShot, mod.rs:201-205)
        NON-parallel_sm build (default features = [] — the LEGACY SERIAL
            binary, LibdeflateSingle for <1 GiB): NEVER bench it as the
            product; verify with GZIPPY_DEBUG=1 → path=ParallelSM first.
            → parallel::single_member::decompress_parallel
              (parallel chunk pipeline — see `src/decompress/parallel/`.
               Output STREAMS: bytes flow to the writer as each chunk
               resolves; CRC32 + ISIZE are verified against the gzip
               trailer AFTER the final chunk is written, so a mismatch
               is a terminal Err with partial output already on the
               writer — same as gzip(1); for file output
               `io::decompress_file` then deletes the dest file.
               Counter `MARKER_PIPELINE_RUNS` proves production routing
               called us; see the deletion-trap test in
               src/tests/routing.rs)
        x86_64 (ISA-L available)        → isal_decompress::decompress_gzip_stream
        any arch, data > 1 GiB (no ISA-L) → decompress_single_member_streaming (zlib-ng)
        default                          → decompress_single_member_libdeflate
```

### Compression

```
T1 L0–L3, ISA-L available → backends::isal_compress::compress_gzip_{to_writer,stream_direct}
T1 L1–L5                  → libdeflate one-shot (ratio probe) or flate2 streaming (zlib-ng)
T1 L6–L9                  → flate2 streaming (zlib-ng)
T>1 L6–L9                 → compress::pipelined::PipelinedGzEncoder → single-member output
T>1 L0–L5                 → compress::parallel::ParallelGzEncoder  → "GZ" subfield multi-block
```

**"GZ" subfield**: gzippy's own parallel format (not standard BGZF). Files produced by
`ParallelGzEncoder` carry a "GZ" FEXTRA subfield with per-block size info; decompression
routes them to `bgzf::decompress_bgzf_parallel`. `PipelinedGzEncoder` output is plain
single-member — decompresses on the single-member path.

## Optimization Branches

There is no `src/experiments/`. Every module on `main` is reachable from a
production code path or is a test fixture / supportive script.

To prototype a new path: add the module under the relevant subsystem
(`src/decompress/`, `src/compress/`, etc.), wire a feature-gated or size-gated
call site in the routing table above, and add a strict correctness test (no
silent fallback). When `make ship` confirms the win, lift the gate. When
abandoned, delete the module — `main` does not host dead code.

Two regression tests lock the parallel single-member wiring:

1. `tests::routing::tests::test_single_member_routing_multithread` — runs
   `decompress_single_member(T=4)` on a 24 MiB input and asserts byte-perfect
   output. Covers "parallel path takes the input" and "falls back correctly
   when speculation fails on adversarial chunks."

2. `tests::routing::tests::test_single_member_parallel_not_slower_than_sequential`
   — runs the same fixture at T=1 (sequential) and T=4 (parallel) and asserts
   `parallel_elapsed` stays within a small multiple of `sequential_elapsed`.
   This is the local-CI guard against a parallel-slower-than-sequential
   regression.

## Active port: rapidgzip → gzippy parallel single-member

The parallel single-member path under `src/decompress/parallel/` is an
in-progress port of rapidgzip's chunked-decode architecture. Vendor
source is in `vendor/rapidgzip/librapidarchive/`; port modules cite
vendor `file:line` in their doc comments — treat those citations as the
reference when a change appears to "work" but looks structurally off.

Capture a trace with `GZIPPY_LOG_FILE=/tmp/sm.log` and analyze via
`scripts/parallel_sm_log_summary.py` before claiming a perf change worked.

## Hard-Won Lessons

**What works**: mmap stdin for multi-threaded (zero-copy, +44%), BufWriter for
stdout, direct FFI, BGZF parallel, 1MB streaming buffer, lock-free parallel.

**What doesn't**: mmap for single-threaded (4x slower from page faults!),
larger blocks for L1 (no help), **speculative parallel decode on arm64**
(16x slower on low-redundancy data — block boundaries are rare, most chunks
become all-marker forcing huge sequential re-decodes),
two-pass scan-then-decode, large pre-allocations.

**arm64 single-member**: currently falls through to libdeflate one-shot (fast enough).
Streaming path only for files > 1 GiB. ISA-L is unavailable on arm64; the parallel
single-member path is gated on `isal_decompress::is_available()` so arm64 never
takes it.

## Branch and PR Workflow

**main is protected** — no direct pushes. Every change goes through a PR.
The pre-push hook (auto-installed by `cargo build`) enforces this locally.

```bash
# Start work
git checkout -b fix/my-change     # or feat/, refactor/, chore/, etc.

# Iterate (same as before)
GZIPPY_DEBUG=1 gzippy -d -c testfile.gz > /dev/null
make
cargo test --release

# Ship
git push origin fix/my-change
gh pr create --fill               # title + body from commit messages
# CI runs → merge when green
```

Emergency bypass (never for performance-affecting changes):
```bash
git push --no-verify origin fix/my-change   # skip local hook only
```

Releasing:
```bash
git tag v0.X.Y && git push origin v0.X.Y   # triggers Release workflow
# Release workflow creates a formula-update PR and auto-merges it
```

## Iteration Loop

```bash
# 1. Make one focused change
# 2. Check routing
GZIPPY_DEBUG=1 gzippy -d -c testfile.gz > /dev/null   # Shows which path is taken

# 3. Local sanity (30s) — catches catastrophic regressions
make

# 4. Correctness
cargo test --release

# 5. Authoritative numbers — only after make passes
make ship   # SSH to neurotic homelab, runs gzippy-dev bench
```

`make route-check` — generates 1MB+10MB test files and shows routing + timing
vs pigz for all four combos (T1/T4 × 1MB/10MB). Use this before ANY decompression change.

## When a tool errors, FIND OUT WHY (user-set 2026-05-31) — applies to every agent

**If a tool call errors, diagnose the cause before doing anything else. Never
retry the same call, and never proceed past it, until you know why it failed.**
This is the governing rule; the specific tactics below are just consequences:

- A *repeated identical* error (e.g. `Cancelled: parallel tool call …`) is the
  signature of an already-wedged channel or an unmet precondition — STOP and
  investigate the FIRST failure, do not loop. Two background agents this campaign
  burned their whole budget retrying through error #1: one on a FULL DISK
  (`cargo build` exhausted the volume → write errors), one on a HUNG
  `python3 -c "<multi-line>"`/heredoc through the Bash tool (broken quoting →
  interactive Python reading stdin → never returns → wedges the tool channel →
  every later call reports "Cancelled"). Both were one `df -h` / one read-the-
  error away from a 30-second diagnosis.
- Consequence: don't run multi-line Python through Bash — Write a `.py` file and
  run `python3 file.py`. Wrap any potentially-hanging command in `timeout`.
  Check `df -h` before/around big builds. But these are downstream of the rule:
  *read the actual error, find the cause, fix the cause.*

## Key Files

| File | Role |
|------|------|
| `src/decompress/mod.rs` | Decompression entry, format detect, routing |
| `src/decompress/bgzf.rs` | gzippy-parallel + multi-member parallel (core engine) |
| `src/decompress/scan_inflate.rs` | Streaming scan-and-inflate path |
| `src/decompress/parallel/single_member.rs` | Parallel single-member decode — entry point |
| `src/decompress/parallel/{blockfinder_validation,gzip_block_finder,async_block_finder}.rs` | Block-boundary finders (validators / offset partitioner / async coordinator) |
| `src/decompress/parallel/{bit_reader,chunk_decode}.rs` | Shared bit reader (rg `core/BitReader.hpp`) / per-chunk decode driver (rg `GzipChunkFetcher::decodeChunk*`) |
| `src/decompress/inflate/consume_first_decode.rs` | Pure-Rust inflate (production helpers used by `bgzf`, `scan_inflate`) |
| `src/decompress/inflate/{consume_first_table,jit_decode,libdeflate_decode,libdeflate_entry,specialized_decode,vector_huffman,double_literal,bmi2}.rs` | Huffman/inflate building blocks |
| `src/decompress/{combined_lut,inflate_tables,packed_lut,simd_copy,simd_huffman,two_level_table}.rs` | SIMD + LUT primitives shared with bgzf |
| `src/backends/isal_decompress.rs` | ISA-L streaming inflate (x86_64 production path) |
| `src/backends/inflate_bit.rs` | Universal inflate-from-bit (ISA-L on x86_64, libz-ng elsewhere) |
| `src/backends/libdeflate.rs` | libdeflate FFI — `gzip_decompress_ex` |
| `src/compress/mod.rs` | Compression entry, routing |
| `src/compress/parallel.rs` | ParallelGzEncoder — T>1 L0–L5, "GZ" multi-block output |
| `src/compress/pipelined.rs` | PipelinedGzEncoder — T>1 L6–L9, single-member output |
| `src/backends/isal_compress.rs` | ISA-L compression (x86_64 T1 L0–L3) |
