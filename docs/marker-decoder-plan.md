# v0.6 Marker-Based Parallel Single-Member Decompressor — Plan

This document tracks the multi-PR rollout of the v0.6 marker pipeline that
replaces v0.5.1's speculative-window two-pass design. It exists per premortem
mitigation D2: a checked-in milestone document means slippage shows up here,
not in nebulous "we'll get to it." Every PR below has a clear standalone
success criterion. Each is independently revertable.

## Premortem context

PR #90 introduced v0.5.1, a correct two-pass speculative-window decoder. It
ships ~2N total compute work and at T=4 on CI's 4-physical-core ubuntu-latest
runner produces 288 MB/s vs. rapidgzip's 327 MB/s (0.88×) — below the 0.99
threshold the project wants. The fix is rapidgzip-style marker-based
parallelism, which has ~1.1N total compute work (decode + cheap marker
resolution).

Prior marker attempts (b3bf6df, 4bbf04f predecessor) failed because:
1. The pure-Rust marker decoder was 22 MB/s — slower than sequential ISA-L.
2. It crashed on non-byte-aligned chunk starts.
3. It lived outside the production path and got cleanup-deleted.

The premortem lives at `docs/marker-decoder-premortem.md`. It names every
failure mode (F1–F6, plus the A/B/C/D/E mitigation catalogue) and the
defense that ships *together with* the code at risk.

## Rollout

### PR #93 — `fast_marker_inflate.rs` (correctness only) — **OPEN**

Chains off PR #90. Adds the marker-emitting deflate decoder; **does not** wire
into production routing. Lands when CI is green on this PR's own tests.

- [x] Premortem mitigation A1 — u16-vs-u8 output bandwidth synthetic
      benchmark. Result: ratio 1.03 on arm64. (commit c69c52f)
- [x] `fast_marker_inflate::decode_chunk_markers(data, start_bit_offset) -> (Vec<u16>, end_bit)`.
      Pure Rust, ~600 lines including tests. Reuses
      `inflate::consume_first_decode::Bits`. Canonical-Huffman table built
      locally. (commit d3205ea)
- [x] Differential fuzz harness: 200 trials, random inputs up to 16 KiB,
      random compression levels, byte-equal against
      `inflate_consume_first` oracle. (commit d3205ea)
- [x] Corner-case fixture tests: cross-chunk back-ref markers, boundary
      spanning, RLE distance-1, bit offset 0..=7 starts. (commits d3205ea,
      e59a64a)
- [x] End-to-end integration test: split a real deflate stream at a real
      mid-stream block boundary, decode the suffix with markers, resolve
      against the prefix's last 32 KB, verify byte-perfect. Selected split
      is non-byte-aligned in practice (bit offset within byte). 60K+
      markers resolved correctly. (commit e59a64a)
- [x] Throughput sanity measurement: ~1352 MB/s/thread on arm64 Mac vs.
      production `inflate_consume_first` ~23200 MB/s. Per-thread is 17×
      slower than the tuned production u8 decoder, but absolute is 4× the
      rapidgzip target per thread; T=4 saturates memory bandwidth. (commit
      d3205ea, `#[ignore]` throughput_vs_oracle test)

### PR #94 — wire pipeline into production, delete v0.5.1 + legacy MarkerDecoder

Lands when PR #93 is merged. The hard part is mitigation C — *make the marker
pipeline the only parallel single-member path so future cleanups can't delete
it*. Concretely:

- [ ] Rewire `decompress::parallel::single_member::decompress_parallel`:
      - Phase 1 (parallel workers): `fast_marker_inflate::decode_chunk_markers`
        on each chunk's bit range. Output: `Vec<u16>` per chunk.
      - Phase 2 (sequential or pipelined): for each chunk i, take chunk i−1's
        last 32 KB (post-resolve) as the window, call `replace_markers`,
        convert u16 → u8 via `u16_to_u8` which fails fast on leftover markers,
        write to the output Vec.
      - Phase 3: CRC32-verify against gzip trailer; write to writer.
- [ ] Delete the v0.5.1 speculative-window code (phase1_decode_parallel,
      phase2_finalize_parallel, all related types). Same file.
- [ ] Delete the legacy `marker_decode::MarkerDecoder` and the supporting
      `try_decode_chunk` (now unused). Keep only `skip_gzip_header` if
      anything depends on it.
- [ ] Drop the `MIN_PHYSICAL_CORES_FOR_PARALLEL = 8` gate in
      `decompress_single_member`. Marker pipeline at T=2 already matches
      sequential at ~1.1N total work; at T≥3 it wins.
- [ ] Ratchet `single_member_vs_rapidgzip` to 0.99 and
      `single_member_vs_pigz` to 1.0 in `scripts/check_guards.py` and
      `scripts/benchmark_single_member.py`.
- [ ] **The deletion-trap killer test**: a routing-level assertion that on
      x86_64 with ISA-L and `num_threads ≥ 2`, the marker pipeline is what
      runs (not a silent fallback). Implementation: a thread-local atomic
      counter incremented inside the marker pipeline; the test reads it
      before and after the decode and asserts an increase. Without this,
      output-equivalence tests pass even if `decompress_single_member`
      silently falls back to sequential ISA-L, and the marker pipeline goes
      uncovered → regressions land → future cleanup deletes it.
- [ ] Pin the rapidgzip submodule revision so the threshold goalposts stop
      moving.
- [ ] Update `CLAUDE.md` routing table.
- [ ] Update `RATIONALE.md` (new file in `src/decompress/parallel/`)
      explaining the v0.6 design and pointing at this plan + the premortem.
      A future agent doing cleanup reads this before deleting anything.

**Acceptance**: CI Single-Member Decomp Tmax (x86_64) passes
`single_member_vs_rapidgzip ≥ 0.99` on Silesia. The routing-assertion test
is green. The full test suite is green.

### PR #95 (optional) — SIMD inner-loop optimizations

Lands only when PR #94 is merged and **only if `make ship` on the homelab
shows headroom**. The current decoder is unoptimized (no BMI2/AVX2 inner-loop
tricks). At 1352 MB/s/thread it already comfortably wins on 4-core CI; on a
16-physical-core homelab it may want a SIMD fast path.

- [ ] AVX2 vectorized literal-run emission (8 u16s per cycle).
- [ ] BMI2 BZHI for bit-buffer extraction.
- [ ] Double-literal table lookup (two symbols per peek).
- [ ] Re-measure on `make ship`. Land only if measurable.

## Decision log

- **2026-05-12**: u16 bandwidth penalty empirically retired (ratio 1.03 on
  arm64). No design pivot needed.
- **2026-05-12**: PR #93 lands with 12 tests including 200-trial diff fuzz
  + mid-stream integration. Throughput 1352 MB/s/thread on arm64.
- **TBD**: PR #94 ratio-vs-rapidgzip on CI to be measured. Target: ≥ 0.99.
