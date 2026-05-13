# Why `src/decompress/parallel/` exists, and what NOT to delete

Read this first if you are about to delete code in this directory.

## What this module does

`parallel::single_member::decompress_parallel` is the production parallel
single-member gzip decoder. It is called by
`crate::decompress::decompress_single_member` when ISA-L is available
(`x86_64` builds with the `isal-compression` feature) and the compressed
stream exceeds 10 MiB at `num_threads > 1`. It is **the** parallel decoder for
the most common gzip case (`gzip -c file > file.gz` produces a single-member
stream). The CLI `-p N` flag selects parallelism here.

## Why it is structured the way it is

The current design is the **v0.6 marker pipeline**:

1. `fast_marker_inflate::decode_chunk_markers_bounded` — pure-Rust deflate
   decoder, parallelizable, emits `Vec<u16>` with markers (`≥ MARKER_BASE`)
   for back-references that reach into the preceding chunk's 32 KB window.
2. `replace_markers::replace_markers` — SIMD-vectorized marker resolution
   (AVX2 / NEON / scalar) that runs sequentially across chunks, using each
   chunk's last 32 KB as the window for the next.
3. `single_member::phase2_resolve_sequential` — drives phase 2, converts
   each chunk u16→u8, accumulates output, computes CRC32 incrementally.
4. `single_member::decompress_parallel` — verifies CRC + ISIZE against the
   gzip trailer **before** writing to the output writer. Failed verification
   returns `Err(...)` with the writer untouched, the caller falls back to
   sequential ISA-L.

This shape was not arrived at casually. It is the surviving design after
prior attempts each got removed during cleanup:

- `b3bf6df` (2025) — first MarkerDecoder. Pure-Rust at **22 MB/s**, slower
  than sequential ISA-L. Decoded bit-by-bit, cloned its input on
  construction, no canonical-Huffman lookup table.
- `4bbf04f` (Feb 2026, v0.3.0) — abandoned markers entirely; switched to
  ISA-L inflatePrime. Commit message: *"Root cause of all prior failures:
  pure-Rust marker decoder at ~22 MB/s/thread; marker_decode.rs always
  returned None because it requires byte-aligned input."* Two killers
  stacked — too slow AND wrong API for the production bit-offset chunk
  starts.
- `3eba641` (Feb 2026) — cleanup deleted **9,300 lines** across
  `hyper_parallel.rs`, `simd_parallel_decode.rs`, `parallel_decompress.rs`,
  `parallel_inflate.rs`, `ultra_decompress.rs` as "dead experimental
  modules." None were wired into the production CLI; nothing called them;
  they looked dead. The good SIMD work (`replace_markers_avx2`, etc.) was
  lost along with the experiments.
- v0.3.0 (Feb 2026) — `inflatePrime` re-decode design had a fatal bug: phase 1
  stored only `(start_bit, end_bit)` and discarded the decoded bytes; phase 2
  re-decoded the entire stream sequentially. **1.75× *slower* than sequential
  ISA-L**. The CI guards at the time were absolute (vs rapidgzip/pigz) without
  a "parallel ≥ sequential" floor, so it landed.
- v0.5.1 (May 2026) — fixed the v0.3.0 bug with a correct two-pass
  speculative-window design. Correct, but did **2N total compute work** —
  on a 4-physical-core CI runner produced 288 MB/s vs. rapidgzip's 327 MB/s
  (0.88×), below the 0.99 target. The marker pipeline replaced it.

## The deletion-trap killer

Every prior marker attempt that didn't ship as production code died in
cleanup because **output-equivalence tests don't fail when the marker
pipeline is silently bypassed**. If `decompress_single_member`'s routing
gate fails (e.g., someone tightens a condition by accident) and the call
falls through to sequential ISA-L, all output tests still pass. The marker
pipeline becomes uncovered, regressions land in adjacent code, and the
next cleanup removes it because "nothing tests this."

The defense is `MARKER_PIPELINE_RUNS` — a process-global atomic counter
incremented at the successful tail of `decompress_parallel`. The test
`src/tests/routing.rs::test_marker_pipeline_actually_runs_on_x86_64_isal`
snapshots it around a real `decompress_single_member(T=4)` call and asserts
it incremented. On x86_64 + ISA-L (the only target where the pipeline
should fire), failure to increment means the routing gate is broken — the
test goes red even if output is byte-perfect.

**Do not delete this counter or that test without replacing it with an
equivalent assertion.** They are not redundant with output tests.

## What you may not delete without thinking

| Symbol | Role | Why deletion is dangerous |
|---|---|---|
| `parallel::single_member::decompress_parallel` | Production entry | This IS the parallel single-member path. |
| `parallel::single_member::MARKER_PIPELINE_RUNS` | Routing-assertion counter | The deletion-trap killer. |
| `parallel::fast_marker_inflate::decode_chunk_markers_bounded` | Marker decode | Phase 1 of the pipeline. |
| `parallel::fast_marker_inflate::record_block_starts` | Truth-source for block boundaries | Test-only but used by the integration test. |
| `parallel::replace_markers::replace_markers` + `MARKER_BASE` + `u16_to_u8` | Phase 2 | SIMD resolution. |
| `tests::routing::test_marker_pipeline_actually_runs_on_x86_64_isal` | The killer test | See above. |
| `tests::routing::test_single_member_parallel_not_slower_than_sequential` | v0.3.0-class regression guard | Catches the 1.75× regression at PR time. |

## What you may delete (with a follow-up PR)

`parallel::marker_decode::MarkerDecoder` (the legacy ~22 MB/s pure-Rust
decoder) is no longer used by production. It currently exists only because
test fixtures in `src/tests/correctness.rs` reference it. A follow-up PR
should either (a) retarget those tests at `fast_marker_inflate` or (b)
delete them as redundant with `fast_marker_inflate`'s 200-trial
differential fuzz against `inflate_consume_first`. Then delete
`MarkerDecoder` and most of `marker_decode.rs`, keeping only
`skip_gzip_header` (which is used by both `fast_marker_inflate` and
`single_member`).

## Where the plan lives

`docs/marker-decoder-plan.md` is the multi-PR rollout document. It records
which PR added what, the throughput measurements that justified the
design, and the explicit decision log. If you are about to change anything
here, update that document so future agents see the current state.

The premortem that informed the design lives at
`docs/marker-decoder-premortem.md` — checked into the repo so a future
agent doesn't have to spelunk through PR comments. The single most
important insight: **prior attempts were deleted because they lived
outside the production path**, not because they were algorithmically
wrong. Keep this module wired in and tested, and it stays.
