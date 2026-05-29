# gzippy decoder plan — ONE pure-Rust DEFLATE engine (canonical v2, 2026-05-29)

v2 incorporates a 3-angle Opus review (direct critique + premortem + full
architectural planning). Supersedes prior plans (git history keeps them).

## THE DELIVERABLE

> gzippy ships ONE pure-Rust DEFLATE engine as its only decode path —
> byte-exact on a continuous fuzz corpus, ≥ libdeflate/ISA-L/zlib-ng/rapidgzip
> on every **decoder-closable** cell (structural cells justified, see done-test
> #5), with `isal-rs`/`libdeflate-sys`/`zlib-ng` removed from the production
> dep graph (kept behind a `dev`/`oracle` feature as permanent CI oracles).

### Done-test (binary, checkable)
1. **One engine.** `Inflate<M: DecodeMode, O: OutputModel>` (Clean-u8 /
   Markers-u16 × Owned / Streaming) with ONE shared `decode_huffman_body<S: OutputSink>`,
   one `Bits`, one header parser. `deflate_block::Block`, `consume_first_decode`
   loops, `isal_lut_bulk`, the LUT graveyard DELETED.
2. **Sole path.** Every format routes to `Inflate`; no FFI inflate on a
   production path **on a platform that has flipped** (per-platform, done-test #5b).
3. **C gone (per-platform).** `isal-rs`/`libdeflate-sys`/`zlib-ng` only behind
   `dev`/`oracle`; a platform's prod dep graph drops them only after that
   platform shows measured ≥parity.
4. **Correct.** lib tests + **continuous cargo-fuzz (dual-oracle: flate2 AND
   libdeflate, adversarial seed corpus)** + silesia/multimember/bgzf/>1GiB-stream/
   seekable-index all byte-exact. C oracles diffed in CI forever.
5. **Fast — against a PRE-CLASSIFIED cell list (no self-judged waivers):**
   - **Closable (must hit literal ≥parity):** single-member T1, single-member
     T2–T8, multi-member all-T, bgzf all-T, incompressible, L9.
   - **Structural (waiver allowed, with a written measured TMA/attribution
     justification; a NEW waiver needs adversary-advisor sign-off, not self):**
     tiny-file (~300 µs irreducible Rust process startup); single-member T9–T16
     on an 8-P-core box (hardware core count); and **single-member T4–T8 vs
     rapidgzip — CONFIRMED STRUCTURAL by M0.3 (2026-05-29, interleaved best-of-15
     T8): pure 1709 / isal 1818 / rapidgzip 2483. Only 14.1% of the
     pure→rapidgzip gap is inner-loop (pure→isal, closable by collapse+A2);
     85.9% is isal→rapidgzip = pipeline/memory/parallel-scaling, NOT
     decoder-closable.** ⇒ the collapse does NOT make gzippy beat rapidgzip at
     T4–T8; it closes the 14% inner-loop slice (matching ISA-L, enabling C
     deletion). Beating rapidgzip here requires a SEPARATE parallel-pipeline /
     buffer-lifecycle / allocator re-port (out of the decoder's scope; a future
     deliverable, not this one).

## M0 — MEASUREMENT & ADVISOR-FEEDBACK STRATEGY (do FIRST; gates everything)

Every later phase is validated by these; establishing them first is the
highest-leverage work (per the review, most failures are measurement
self-deception or unguarded regressions).

**Wall perf:** ONLY interleaved-delta A/B (both binaries one trial loop, share
contention, report the DELTA; absolute is meaningless — box has ~15-20% jitter).
Freeze LXC 111+105, pin physical P-cores `0,2,4,6,8,10,12,14`, turbo ON.
**No wall number enters a commit message unless it's an interleaved-delta, n≥5.**

**Inner-loop perf:** deterministic `inner_bench` instructions/byte (jitter-immune):
modes `pure` / `consume_first` / `libdeflate` / `bootstrap` (marker path). Build
the **`markers` mode FIRST** for E3 (measures `Inflate<Markers>` in isolation).

**Codegen-diff guard (NEW, the architect's #1 containment):** a `make`/CI check
that disassembles `Inflate<Clean,Owned>::decode_huffman_body` and asserts ZERO
`call` in the FASTLOOP region + instruction count within ±2% of a checked-in
golden. Converts a silent trait-inline deopt of the clean path into a hard
every-commit failure. Live from the first trait commit (E3.2).

**Dual-arch CI gate (NEW, kills the cfg trap that bit twice):** CI builds AND
runs the corpus test for BOTH `x86_64` and `aarch64` (`--no-default-features
--features pure-rust-inflate`) every commit on this branch. An arch-only break
becomes loud, not "caught only on the box."

**Pre-registered kill-numbers (NEW):** before writing a lever's code, write its
abandon threshold. E3: "abandon the unified-markers path if, after the first
complete `RingSink` prototype, `inner_bench markers` is not ≤ [13 + measured
u16/ring/scan floor from M0.3] AND interleaved-delta T8 is not ≥ +[X]%."

**No same-session win-banking on irreversible deletes:** a perf win that
precedes deleting a decoder (e.g. `deflate_block`) must be re-confirmed in a
SECOND independent run before the delete commit.

**M0.3 — TMA attribution of the rapidgzip T8/T16 gap (NEW, before E3):** perf
TMA on gzippy-pure vs rapidgzip at T8 and T16; state what fraction is
inner-loop (decoder-closable) vs backend-memory/parallel-scaling (structural).
This decides whether T4–T8-vs-rapidgzip is a closable cell or a structural
waiver (done-test #5). Cheap; do before betting E3 on it.

**Advisor-feedback strategy (process rule, mechanized):** consult an Opus
advisor (Agent `claude`) BEFORE: each design fork, each claimed-done, each
surprise, and each irreversible delete. Specifically — adversary sign-off
required for: any new structural-cell waiver; the E3 kill-number; deleting
`deflate_block`/C deps. Use multi-angle (critique + premortem + architecture)
for plan-level decisions, single adversary for lever-level.

## Architecture (from the full architectural-planning pass)

Collapse the two inner loops (`resumable.rs` FASTLOOP ~13 ins/byte +
`deflate_block.rs` marker loop 27.4 ins/byte) into ONE generic body:

```rust
fn decode_huffman_body<S: OutputSink>(bits, sink: &mut S, litlen, dist, marker: &mut S::Marker) -> ...
```

- `DecodeMode`: `Clean{Elem=u8, Sink=LinearSink, Marker=NoMarkers, EMITS_MARKERS=false}`
  vs `Markers{Elem=u16, Sink=RingSink, Marker=RingMarkers, EMITS_MARKERS=true}`.
- `OutputSink` (sealed, 2 impls): `put_literal`, `copy_match` (the divergence —
  LinearSink = linear+overshoot+AVX; RingSink = ring-modulo + pre-init marker
  zone), `headroom`, `position`. `#[inline(always)]` over ZST/pointer receivers;
  NO `dyn`; split `copy_match` into `#[inline(always)]` hot + `#[inline(never)]`
  cold (window/wrap/RLE) so the hot path always inlines.
- `MarkerPolicy`: `NoMarkers` (ZST, empty bodies → vanish) / `RingMarkers`
  (the `distance_to_last_marker_byte` + backward scan). Gated by `const ACTIVE`.
- `OutputModel`: `Owned{PER_ITER_YIELD_CHECKS=false}` (one-shot, no resumable
  tax) / `Streaming{=true}` (yield-on-fill for the parallel-SM wrapper).
- **DROP `ArchProfile` as a type axis** — SIMD lives in 2 runtime-dispatched
  leaf helpers (bit-extract, match-copy), not a monomorphization axis.
- Approach **(a)-via-trait**, NOT (b)-bolt-markers-onto-ResumableInflate2 (that
  drags the SlidingWindow slow path into the marker case — wrong dependency).

## Current state (committed, reimplement-isa-l)
0c consolidate 2 FASTLOOPs→1 (`d15e35e`, +7%); A2 subtract-elim (`d4c9294`,
−2.5% ins/byte); bootstrap instrumentation refuting B1 (`ed4ca13`); `inner_bench
bootstrap` (`e68be90`, marker path = 27.4 ins/byte); E2 marker-fuzz seed
(`e630f01`, 200 cases — to be hardened into continuous dual-oracle fuzz);
plan v2 (this). Interleaved-delta + inner_bench harness built; `vector_huffman` deleted.

## Execution order (re-sequenced per the review)

- **M0** — measurement & advisor strategy above: build the codegen-diff guard +
  dual-arch CI + `markers` inner_bench mode + run M0.3 TMA attribution +
  pre-classify cells. FIRST.
- **E2** — harden the fuzz harness into continuous cargo-fuzz, dual-oracle
  (flate2+libdeflate), adversarial seed corpus (marker/clean straddle,
  max-distance back-refs, degenerate Huffman, stored boundaries, empty dynamic
  blocks), + snapshot `Block` golden vectors. NO E3 code until this exists.
- **E1** — delete dead code (after E2's snapshot capability exists, so nothing
  deletable is lost): `double_literal` → bgzf dead decode path → orphaned LUT modules.
- **E3** — THE COLLAPSE via `OutputSink` (E3.1 extract FASTLOOP to generic body,
  flat; E3.2 traits+LinearSink, codegen-guard live; E3.3 RingSink unit-tested;
  E3.4 env-gated wire-in, fuzz+inner_bench; E3.5 bank interleaved-delta, flip;
  E3.6 delete old marker loop + ISA-L huffman). Kill-number pre-registered.
- **E6 (per-format, decoupled from E4)** — flip each format's default to the
  pure engine the moment it's at parity for that platform, and delete the C dep
  for that platform. "C gone" is the deliverable; don't block it on E4 polish.
- **E5a** — generalize x86 (single-member T1/streaming/>1GiB, multi-member,
  bgzf, seekable-index) through `Inflate` drivers in `drivers.rs`.
- **E5b** — arm64 NEON parity (separate done-test; arm64 correctness is
  immediate scalar Rust, NEON is a perf leaf in `copy_match`).
- **E4 (optional polish)** — close the residual 13→8.6 algorithmic gap + the
  1.20× resumable tax. ONLY after E6 ships. Don't churn micro-levers (project
  record warns); gate every attempt on `inner_bench` + a kill-number.
- **E7** — close/justify the pre-classified cells.

## Top risks (premortem) + baked-in mitigations
1. Collapse ships a silent correctness regression on adversarial input → M0/E2
   continuous dual-oracle fuzz + golden vectors BEFORE any E3 code.
2. Collapse is perf-neutral (27.4→13 wishful) → M0.3 attribution sets a measured
   floor; pre-registered kill-number; markers inner_bench first.
3. Deleting C regresses arm64/Mac (no ISA-L/NEON) → per-platform parity gate on
   C deletion (done-test #3); keep libdeflate as arm64 prod path until E5b proven.
4. "≥ everyone everywhere" never converges / micro-lever churn → pre-classified
   finite cell set; E6 ahead of E4; E4 optional.
5. Measurement self-deception → interleaved-delta-only for wall, no same-session
   banking on deletes, codegen-diff guard.
6. cfg trap (3rd bite) → dual-arch CI build+corpus gate.
7. inner_bench green but wall flat (drain is ~half the marker tax) → E3 success
   gate is interleaved-delta T8 wall, not inner_bench alone.

## Prime directive
Correctness over speed, always. One gated, bisectable commit per step. No
unvalidated decoder change ships. Multi-week effort; pace it.
