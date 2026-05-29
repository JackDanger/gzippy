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

**GATE (must precede any rewrite/delete — genuine technical dependency):**
- Measurement+referee infra: interleaved-delta harness (done), `inner_bench`
  modes incl. a new `markers` mode, the codegen referee, the dual-arch CI gate.
- Continuous dual-oracle cargo-fuzz + adversarial seeds + golden-vector snapshots
  (the current 200-case marker fuzz `e630f01` is the seed — make it continuous +
  dual-oracle).
- M0.3 attribution: DONE (rapidgzip gap classified structural).

**Then all of these run concurrently (no inter-ordering except the bisectable-
commit rule and per-platform parity gates):**
- **Dead-code deletion:** `double_literal` → bgzf dead decode path → orphaned LUT
  modules (`combined_lut`/`packed_lut`/`simd_huffman`/`two_level_table`/
  `ultra_fast_inflate`). Independent of everything else once the fuzz/golden gate
  exists.
- **The decode-loop race** (above): standalone Clean, standalone Marker, shared
  trait; race; ship fastest-correct. Wire the bootstrap (marker) + clean paths to
  the winners; delete `deflate_block::Block`/`consume_first`/`isal_lut_bulk` loops
  the race retires (after golden snapshot + second-run confirmation).
- **All-platform drivers:** route single-member T1/streaming/>1GiB, multi-member,
  bgzf, and arm64 through the pure engine via `drivers.rs`. arm64 scalar
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

## Current state (committed, reimplement-isa-l)
0c consolidate 2 FASTLOOPs→1 (`d15e35e`, +7%); A2 subtract-elim (`d4c9294`,
−2.5% ins/byte); bootstrap instrumentation refuting B1 (`ed4ca13`); `inner_bench
bootstrap` (`e68be90`, marker path 27.4 ins/byte); marker-fuzz seed (`e630f01`);
M0.3 structural classification; `vector_huffman` deleted. Interleaved-delta +
inner_bench harness live.

## Prime directive
Correctness over speed, always — enforced by exhaustive automated verification
(continuous dual-oracle fuzz, dual-arch CI, codegen referee), not by going slow.
