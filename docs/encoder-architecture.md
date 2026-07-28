# Encoder architecture — the target shape

This document exists because it did not. The encoder was being optimized locally, for a
long time, with no written target architecture, and the structure drifted: eight entry
points with two different meanings of "streaming", three matchfinders, seven parse
strategies for thirteen levels, two boolean parameters that must never be confused, and a
600-line content detector the project's own rules forbid.

This is the shape we are building toward. Every refactor should move toward it or be
justified against it.

## The product constraint that drives everything

At the level the user typed, output at least as small AND less wall time than gzip, pigz,
libdeflate-gzip and igzip. Per-label, not curve. That means:

* **Output bytes are a contract, not an implementation detail.** At L2/4/5/6/7 our output
  is byte-identical to libdeflate's. Any restructuring that changes those bytes must be
  measured against every rival at that label before it lands — twice now a change that
  "only cost 0.02%" flipped tied cells to failing.
* **Peak memory is a user-visible axis** even though no benchmark in the matrix measures
  it. A tool that OOMs where gzip worked is not a drop-in replacement.

## Layering

```
encode/
  api/        the only public surface; format + source + sink, nothing else
  config/     LevelProfile: level -> parse family + knobs. One table, no detection.
  types/      EncodeRequest, BlockRole, InputMode, and the other named alternatives
              that replace boolean parameters
  framing/    gzip header/trailer, CRC, stored-block emission, BitWriter.
              Shared by every orchestrator. Knows nothing about parsing.
  t1/         Step 1. The single-threaded kernel and its streaming loop.
  parallel/   Step 2. Segments the input, calls the T1 kernel per segment, orders
              output. Depends on t1; t1 never depends on it.
  ultra/      Step 3. Near-optimal planning. Swaps the parser only, reusing framing.
```

### "Three separate paths" means isolation, not duplication

The owner's requirement is that later work cannot regress earlier work. That is a
DEPENDENCY constraint, not a mandate for three copies of the encoder:

* three orchestrators (`t1`, `parallel`, `ultra`), one shared framing and config substrate
* `parallel` calls the `t1` kernel per segment — it must never fork parser logic
* the dependency edge points one way: `parallel -> t1`, `ultra -> framing`, never reverse
* the isolation is enforced by gates, not by copying: a T1 per-label cell that passes must
  keep passing when `parallel` or `ultra` changes

Three copies of a parser is how you get three subtly different answers to the same
question, which is the failure this rule exists to prevent, not the rule itself.

## Naming

One scheme for entry points: `encode_<format>_<source>_to_<sink>`.

| name | meaning |
|---|---|
| `encode_gzip_bytes_to_vec` | whole input in memory, gzip framing |
| `encode_gzip_reader_to_writer` | single pass, bounded memory, gzip framing |
| `encode_deflate_bytes_to_vec` | raw DEFLATE, no framing |
| `encode_deflate_segment_to_sink` | internal: one segment of a concatenated stream (T>1) |

Rules learned the hard way:

* **"streaming" is banned as a bare adjective.** It meant two unrelated things —
  single-pass I/O, and chunk-concatenation for the parallel path. Say `reader_to_writer`
  or `segment`.
* **No implementation invariants in public names.** `compress_gzip_padded` leaked a
  matchfinder slack requirement into the API. If a caller must supply slack, that belongs
  in an internal name and a type, not a public one.
* **No boolean parameters where the two values are not obviously opposites.**
  `is_last` (mark BFINAL) and `consume_all` (this buffer will be refilled) are
  independent, and conflating them silently truncated the parallel path. They are now
  `BlockRole::{Interior, Final}` and `InputMode::{Drain, Bounded}`. A reader should not
  have to remember which `true` is which.

## Minimal strategy set

Seven parse strategies and three matchfinders for thirteen levels is accumulated debt.
Target:

| parse family | levels | matchfinder |
|---|---|---|
| `Store` | 0 | none |
| `Fast` | 1-3 | hash chain |
| `Lazy` | 4-9 | hash chain |
| `NearOptimal` | 10-12 | binary tree |

`Greedy`, `Lazy2` and `Fast0` are policy fragments — a lazy-depth knob and a scan-step
ramp — and belong in `LevelProfile` as parameters, not as top-level strategy variants that
each fork the hot loop. A matchfinder or strategy that does not own at least one winning
cell should be deleted, not kept for symmetry.

## Where the wall gap lives

At L6 the phase decomposition is `parse_match` 94.4%, `huffman_encode` 3.8%, everything
else under 1%. Our output at L2/4/5/6/7 is byte-identical to libdeflate's, so the 2-10%
wall gap is not algorithmic — it is instruction-path shape. The structural suspects, in
order:

1. strategy indirection and generic control flow inside the hot loop
2. abstraction crossings between "find match", "score it" and "emit token"
3. matchfinder memory layout and prefetch cadence
4. bounds checks and panic paths the optimizer could not remove

The architecture serves this directly: a level-specialized, monomorphic kernel chosen ONCE
at entry, with no per-position branch on strategy, is the shape that lets the inner loop be
tuned at all. That is why the architecture cut comes before the micro-tuning, not after —
tuning a loop whose control flow is about to change is wasted work.

## What is deleted, and why it is not "cleanup"

* `parse/gated.rs` and every `GZIPPY_L1TUNE_*` / `GZIPPY_L3TUNE_*` env var — the project
  forbids content detection and environment knobs in the production path. This is a
  correctness-of-contract item, not tidying.
* public `compress_gzip_padded` / `deflate_padded_in_place` — leaked invariants.
* the third matchfinder, and any strategy that owns no winning cell.
* one of `parallel.rs` / `pipelined.rs` — there must be exactly one parallel orchestrator.
