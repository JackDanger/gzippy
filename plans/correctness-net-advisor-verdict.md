# Correctness-net hardening — disproof-advisor verdict (2026-06-07)

Standing differential/seam/proptest/fuzz net built ahead of the copy-free-to-final
clean-tail refactor + inner-Huffman rate work. Independent Opus disproof advisor was
asked: would the net actually CATCH each plausible copy-free-to-final break?

## Verdict (as delivered) + what was done in response

**(a) flip-seam off-by-one — CATCH (gap closed).**
`seam_crossing` asserts full byte-equality with back-refs spanning the seam into the
oldest pre-flip window; a 1-byte window placement error shifts every copied byte →
mismatch. Advisor gap: an off-by-one only manifests if a back-ref touches the EXTREME
window byte. RESPONSE: added `seam_distance_exactly_window_length` (period == 32768, the
max DEFLATE distance == window length) and `seam_distance_one_under_window` (32767) so
both window extremities are read deterministically.

**(b) post-flip back-ref reads UNINITIALIZED reserved tail — was a probabilistic MISS;
now a deterministic CATCH.**
Advisor: the reserve is genuinely uninitialized; a read-before-write reads garbage that a
differential catches only *usually* (zeroed pages can read as the correct byte) — and it's
UB regardless. RESPONSE: added an OPT-IN test-only reserve poison
(`GZIPPY_POISON_RESERVE`, `#[cfg(test)]` only, no-op in production / off by default so it
never perturbs a timing gate) that fills the reserved spare with `0xCD` in
`segmented_buffer::{writable_tail,writable_tail_reserve}`. New test
`seam_poisoned_reserve_no_read_before_write` runs the fold path under the poison on a
dedicated thread (sub-16-MiB and >16-MiB-regrow tails) → any read-before-write
deterministically corrupts. PASSES → the production fold path writes-before-read across
the flip seam and across the clamp regrow. (Advisor also suggested one MIRI/ASAN run;
recorded as a follow-up — the poison gives the deterministic catch without it.)

**(c) >16 MiB reserve-clamp mis-size — was a MISS; now covered.**
`RESERVE_CLAMP = 16 MiB` in `gzip_chunk.rs:1001` (reserve = compressed*8 + 1 MiB, clamped).
Advisor: the seam tests' ~256 KiB tails never reach 16 MiB, and the e2e chunk size is
partition-driven (incidental, never targets the boundary). RESPONSE: added
`seam_reserve_clamp_just_under_16mib` (15.7 MiB clean tail → clamp inactive) and
`seam_reserve_clamp_over_16mib` (25.1 MiB clean tail → reserve under-sizes → amortized
regrow; post-flip back-refs read across the realloc). Both assert byte-equality vs the
straight-decode slice and both run under the poison in (b)'s test.

## Net summary (all additive; local results all green on aarch64 + Rosetta x86_64)
- `src/tests/seam_crossing.rs` — production FOLD path (decode_chunk_window_absent →
  resolve_and_narrow_markers_in_place → merge), synthetic streams; flip + cross-seam
  back-refs (incl. window extremities), consecutive-chunk handoff, 16-MiB clamp straddle,
  poisoned-reserve read-before-write. Oracle: straight whole-stream decode, cross-checked
  vs flate2.
- `src/tests/diff_multi_oracle.rs` — e2e parallel-SM (forced via MARKER_PIPELINE_RUNS),
  THREE oracles (flate2/zlib-ng, libdeflate FFI, zlib-ng raw FFI) + independent CRC32 +
  ISIZE, across shapes/levels incl. near-max-distance back-refs.
- `src/tests/inflate_proptest.rs` — proptest random/structured/near-max-distance × levels
  0–9 vs libdeflate oracle (single-thread production primitive).
- `src/tests/inflate_fuzz_loop.rs` — seeded deterministic fuzz: in-suite smoke + `#[ignore]`
  full loop (GZIPPY_FUZZ_SEED/ITERS), differential vs libdeflate.

## Residual / follow-ups
- One MIRI/ASAN pass over the seam net would upgrade (b) from "deterministic via poison"
  to "guaranteed by the checker" (advisor's optional add).
- The poison is opt-in to protect the perf-timed `diff_ratio` gate; the standing CI seam
  run should export `GZIPPY_POISON_RESERVE=1`.
