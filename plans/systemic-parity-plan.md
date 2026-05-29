# Systemic plan: pure-Rust gzippy → rapidgzip parity

Decisive measurement (frozen best-of-6, T8 silesia-large, MB/s) that grounds this plan:
`isal clean-window 2035 ≈ rapidgzip 2067` | `pure clean-window 1626` | `isal normal 1522` |
`pure normal 1425`. The clean-window oracle (`GZIPPY_CLEAN_WINDOW_ORACLE=1`) is the instrument.

## Root cause (one, not two)
gzippy's parallel PIPELINE is already at rapidgzip parity (oracle 2035≈2067 — segmentation/
allocator levers were FALSIFIED, not the target; do NOT reopen them). The gap is that gzippy
has **three** inner decoders: clean (ResumableInflate2), the window-absent **bootstrap**
(`deflate_block::Block`, u16 markers, ~193 MB/s — slow pure-Rust **in both builds**), and the
consumer's `apply_window`. rapidgzip's bootstrap is ISA-L-fast, so its ≤32 KiB marker prefix
is ~free; gzippy's is 10× slower and sits on the consumer's serial path. **One fast pure-Rust
decoder used for BOTH clean and bootstrap collapses both the inner-loop gap and the
"speculation tax" together.**

## Path (dependency-ordered, by leverage). Measure each via clean-window oracle + normal split.

- **P0 — size the bootstrap fraction (½ day, gates P1/P2 order).** GZIPPY_VERBOSE already
  reports bootstrap body_bytes/body_rate; add consumer-wait + copy-volume. If bootstrap is
  the dominant wall drag (expected: ~31% of bytes @ 193 MB/s), proceed P1→P2.

- **P1 — ONE ISA-L-parity pure-Rust clean decoder (HIGHEST leverage; 1626→~2000).** Work in
  `src/decompress/inflate/` against `examples/inner_bench.rs` (instructions/byte, jitter-immune,
  runs locally via Rosetta). (1) Kill the resumable-contract tax in the clean path — non-resumable
  fastloop when FASTLOOP_OUTPUT_MARGIN headroom exists (extend f01eb74, ~-22% inner instr). (2)
  Port libdeflate per-symbol: BMI2 PEXT/BZHI (`bmi2.rs`), multi-literal packed writes (re-attempt
  ca52389 — un-bound by CLAUDE.md), fixed-Huffman static table, `_mm_prefetch`. Land only
  inner_bench wins; confirm via pure clean-window oracle.

- **P2 — reuse the P1 decoder for the bootstrap (1425→~1900; depends on P1).** Add a
  marker-emitting entry mode to the fast loop: first ≤32 KiB emits u16 markers, then
  `contains_marker_bytes` flips and the SAME fast loop continues clean u8. Delete
  `deflate_block.rs` + `absorb_isal_tail` (clean tail already in the u8 buffer). Falsification
  check: isal-normal must ALSO jump (shared bootstrap).

- **P3 — consumer marker-serialization trim (small; only if P0 flags it after P1/P2).**
  `chunk_fetcher.rs:1263-1341` predecessor-wait + apply_window on the consumer critical path.

- **P4 — ship per-arch everywhere (non-perf).** Runtime BMI2/NEON dispatch; wire the pure
  decoder as the SOLE decode path on all arches (today it ships nowhere); multi-member/BGZF
  through the same core; delete C-FFI from the decode graph at parity.

## Pre-registered target
pure normal 1425 → ~2000 ≈ rapidgzip 2067 (within ~3%). Falsification: if after P1+P2 pure-normal
does not approach pure-clean, a 4th cost exists (consumer/copy) — P0 will have predicted it.

DO NOT touch pipeline / allocator / segmentation (falsified + oracle-confirmed not the target).
Full advisor analysis: session 2026-05-29. Memory: [[project_split_reachable_2026_05_29]].
