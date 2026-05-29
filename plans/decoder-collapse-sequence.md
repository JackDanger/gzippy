# Decoder Collapse Sequence — many pure-Rust decoders → ONE engine

**User goal:** "collapse one implementation into another until only one remains,
with a plan that does not introduce regressions so significant we can't make
sense of them." Structure-first; correctness non-negotiable; then it must be
fast enough to delete libdeflate/ISA-L/zlib-ng entirely (`plans/pure-rust-everywhere.md`).

## End state (DEFINITION — needs explicit sign-off before Phase 3)

We land on **ONE decode primitive** — one FASTLOOP, one dynamic-header parser,
one `Bits` reader, one symbol-decode body — instantiated as **two
monomorphs**: `Clean` (u8 linear output) and `Markers` (u16 ring, window-absent),
plus a small handoff stitch where a speculative chunk transitions marker→clean.

This is "one engine, two modes," NOT one runtime code path. The u16 marker ring
is a **hard architectural fork** from u8-linear (different addressing: linear+
overshoot vs ring-modulo+marker-zone), inherited from rapidgzip's own design.
Literally-one-code-path would require abandoning the marker mechanism, which is
how cross-chunk back-references are resolved in speculative parallel decode —
not on the table. "Collapse until one remains" = one source of truth for the
Huffman decode logic; the two output modes and the handoff are plumbing.

## Ground truth (from the 2026-05-29 audit; see git for the two full maps)

- **Engine A** `ResumableInflate2` — clean/u8/resumable, mature FASTLOOP. Carries
  TWO complete inner FASTLOOPs today: libdeflate-LUT (default) + ISA-L-LUT
  (env-gated off). Production parallel-SM windowed decode (pure build).
- **Engine B** `deflate_block::Block` — window-absent, u16 marker RING. Bootstrap
  phase of speculative chunks; hands off to a fresh Engine A for the clean tail.
- **Engine C** `consume_first_decode` — one-shot clean. scan_inflate/index + test
  oracle. Holds the shared `Bits` reader. Contains ~7 DEAD inner-loop variants.
- **Engine D** `isal_lut_bulk` — one-shot clean, env-gated off.
- **DEAD** (no prod callers): bgzf CombinedLUT/decode_*_into/
  decompress_single_member_parallel, ultra_fast_inflate, two_level_table,
  simd_huffman, combined_lut, packed_lut, vector_huffman, double_literal,
  route_c_*, dead decode_huffman_* in consume_first.
- Already shared (NOT duplication to fix): `Bits` (all engines), `copy_match_fast`
  (u8, A+C). Genuine dups to collapse: dynamic-header parser (4×), extra-bit
  tables (multiple). The u8 vs u16 match-copy are DIFFERENT algorithms — keep both.

## Regression-isolation strategy (the guardrails)

1. **Primitive-level golden gate, every commit.** A harness that calls the
   inflate primitive DIRECTLY (not the parallel pipeline — avoids the known
   `project_parallel_test_hang` deadlock and isolates inflate perf from
   allocator/pipeline noise). It asserts:
   - byte + CRC identity vs **snapshotted golden output vectors** captured from
     each decoder BEFORE it is deleted (so "diff vs the prior decoder" stays
     runnable after deletion), and
   - a **hard clean-path floor: fastloop ≥ 1623 MB/s** on the pinned pure binary.
   No merge unless both hold.
2. **One semantic change per commit, env-gated default-OFF** where behavior
   changes; flip the default only after A/B proves parity + correctness.
3. **Never delete-then-discover:** delete a decoder only after its replacement is
   proven AND its golden vectors are snapshotted.
4. **Risk increases monotonically** across phases; clarity is maximized early
   (dead-code delete first).
5. Differential oracles kept forever as `dev`/test-only: flate2 + libdeflate.

## The ordered sequence (advisor-reviewed 2026-05-29)

### Phase 0 — foundation
- **0b FIRST: delete the graveyard** (the DEAD list above). Zero *production*
  callers, but NOT leaf-dead — verified 2026-05-29 the LUT modules are wired
  into dead-but-present code, so deletion is DEPENDENCY-ORDERED (delete callers
  before modules, or the build breaks mid-way):
  1. Delete `bgzf.rs`'s dead pure-Rust decode path first: `decode_stored_into` /
     `decode_fixed_into` / `decode_dynamic_into`, `decompress_single_member_parallel`,
     `get_fixed_tables`, and their TESTS (the only callers of `ultra_fast_inflate`
     at bgzf.rs:6597/7281, `CombinedLUT`, `PackedLUT`, `simd_huffman::MultiSymTable`,
     `two_level_table::{FastBits,TurboBits,TwoLevelTable}`).
  2. Delete `consume_first_decode.rs`'s dead inner-loop variants
     (`decode_huffman_cf_vector` → frees `vector_huffman`; the dead
     `double_literal` users at :1933).
  3. THEN delete the now-unreferenced leaf modules: `ultra_fast_inflate`,
     `two_level_table`, `simd_huffman`, `combined_lut`, `packed_lut`,
     `vector_huffman`, `double_literal`, + their `mod` decls in
     `decompress/mod.rs` / `inflate/mod.rs`.
  Done before freezing any baseline so dead code can't perturb layout/icache.
  Gate after EACH sub-step: `cargo build` + `cargo test --release` green +
  production routing unchanged. (Multi-step surgical untangle, not a flat rm —
  needs a fresh context budget to run the build/test cycles to completion.)
- **0a: build the primitive-level golden differential + bench harness** and
  snapshot golden output vectors for A, B, C, D on the corpus (silesia +
  multimember + edge cases). Freeze the clean-path bench baseline on the lean
  binary. This is the fixed yardstick for everything after.
- **0c: settle the table format.** resumable.rs carries TWO FASTLOOPs
  (libdeflate-LUT vs ISA-L-LUT). The shared inner loop can't exist until one is
  deleted — and which one is a MEASURED perf decision, not a default. Bench both
  on the harness; delete the loser's FASTLOOP (~430 lines). Prerequisite for
  Phase 1. (Prior: ISA-L inner is env-gated off; wholesale ISA-L replacement was
  abandoned — likely keep libdeflate-LUT, but measure.)

### Phase 1 — de-dup the genuine shared primitives (mechanical, byte-exact, ±0%)
Extract ONE dynamic-header parser and ONE extra-bit-table module, each used by
all engines, each its own validated commit. **Do NOT build a u8/u16 match-copy
generic** — the linear-overshoot (u8) and ring-modulo+marker-zone (u16) paths
are different algorithms; a generic would deopt the clean fastloop for no gain.

### Phase 2 — collapse the one-shot engines (C, D) into A's Owned mode
Implement `Inflate<Clean,_,Owned>` (one-shot; per-iteration yield-checks compiled
out). Reroute scan_inflate + isal_lut_bulk callers to it. Delete C's decode paths
(keep `Bits`, relocated to a primitives module in Phase 1) and delete D. Low risk
(non-hot-path). Gate: scan/index byte-exact; clean floor holds.

### Phase 3 — collapse the marker engine (B) into A as `Markers` mode — SPLIT
- **3a (collapse, perf-NEUTRAL):** `Inflate<Markers,_,_>` reuses the shared
  header parser + `Bits` + symbol-decode body, but KEEPS the existing
  `emit_backref_ring` u16 logic. Goal: byte-identical markers vs Engine B, ±0%.
  Reroute the bootstrap; keep marker→clean as a two-instance handoff; delete
  `deflate_block::Block`. This is de-dup only.
- **3b (perf, SEPARATE commit — where A1 actually lives):** give `Markers` mode
  its own fastloop and/or the u16→u8 region split (decode no-marker regions as
  u8 linear; only marker-touched window as u16). The collapse merely ENABLES
  this; folding markers in does NOT auto-inherit the FASTLOOP (its speed depends
  on u8-linear overshoot + margin math that don't hold for a u16 ring). Gate it
  independently so a perf regression never rides with a correctness change.

### Phase 4 — generalize to all formats, replacing C (env-gated longest, flip last)
Each: implement on the unified engine, env-gate, A/B vs the C path it replaces,
flip default only at ≥ parity + byte-exact, then delete the C call site.
- 4a single-member T1 + >1 GiB streaming (replaces ISA-L stream, zlib-ng,
  libdeflate single). **Highest ship-risk** (these are today's default ship
  paths, not pure-gated) → keep gated longest.
- 4b multi-member (replaces libdeflate).
- 4c BGZF / "GZ"-parallel one-shot per block (replaces libdeflate FFI; bar is the
  ~6 GB/s independent-block rate — high).
- 4d arm64 / macOS (replaces libdeflate one-shot; needs NEON match-copy).

### Phase 5 — demolish
Delete isal-rs / libdeflate-sys / zlib-ng-as-decoder from Cargo (keep behind a
`dev`/`oracle` feature for differential tests). Flip `pure-rust-inflate` to
default. Update CLAUDE.md routing tables + release.yml feature flags.

## Why this can't produce regressions we can't make sense of
Every step is one semantic change, gated, validated against snapshotted golden
vectors from the exact decoder it replaces, with a hard clean-path perf floor.
A failure bisects to a single merge with a concrete oracle diff. Dead-code and
mechanical de-dup come first (legible, low-risk); the hard marker fold is split
so correctness and perf never share a commit; the shipping C paths flip last.
