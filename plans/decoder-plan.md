# gzippy decoder plan — ONE pure-Rust DEFLATE engine (canonical, 2026-05-29)

Supersedes `decoder-collapse-sequence.md`, `pure-rust-everywhere.md`,
`global-optimum-2026-05-29.md` (kept in git history for the falsification record).

## THE DELIVERABLE (single, with a binary done-test)

> **gzippy ships ONE pure-Rust DEFLATE engine as its only decode path —
> byte-exact on a fuzzed corpus, and at least as fast as
> libdeflate/ISA-L/zlib-ng/rapidgzip on every archive type × thread count —
> with `isal-rs`, `libdeflate-sys`, and `zlib-ng` deleted from the production
> dependency tree.**

"Done" = all five, each checkable:
1. **One engine.** `Inflate<Mode>` with `Clean`(u8) + `Markers`(u16) modes
   sharing ONE FASTLOOP, one `Bits`, one dynamic-header parser. `deflate_block::Block`,
   `consume_first_decode`'s decode loops, `isal_lut_bulk`, and the LUT graveyard
   are DELETED — `grep` finds one decode loop.
2. **Sole path.** `decompress/mod.rs` routes every format (single-member
   T1+parallel, multi-member, BGZF, all arches) to `Inflate`. No FFI inflate
   call remains on a production path.
3. **C gone.** `isal-rs`/`libdeflate-sys`/`zlib-ng` exist only behind a
   `dev`/`oracle` feature for differential tests; the default/release dep graph
   has none.
4. **Correct.** lib tests + a proptest/fuzz corpus + silesia/multimember/bgzf
   all byte-exact vs flate2 AND libdeflate.
5. **Fast.** `tools/bench/matrix.sh` (interleaved-delta harness) shows gzippy ≥
   every competitor on every cell — OR a written, measured justification that the
   residual is STRUCTURAL (tiny-file process startup; T16 on an 8-P-core box),
   with the decoder itself at parity.

## Measurement discipline (hard-won; non-negotiable)

- **Wall:** only INTERLEAVED-delta A/B (both binaries one trial loop, share
  contention, report the delta) — the box has ~15-20% absolute jitter that
  swamps non-interleaved comparisons. Freeze LXC 111+105, pin physical P-cores
  (`0,2,4,6,8,10,12,14`, not `0-15`), turbo ON.
- **Sub-10% / inner-loop:** deterministic `inner_bench` instructions/byte
  (`pure`/`consume_first`/`libdeflate`/`bootstrap` modes) — jitter-immune.
- **Correctness gate every commit:** silesia md5 `c070ed84` + multimember/bgzf
  roundtrips + (once built) the fuzz corpus, vs flate2 AND libdeflate.
- Build the pure engine: `--no-default-features --features pure-rust-inflate`.
  The parallel-SM + deflate_block code is x86_64-gated — **validate on the box**
  (aarch64 silently compiles it out; this trap bit us twice).

## Current state (this session, all committed on reimplement-isa-l)

- `0c` consolidated 2 FASTLOOPs → 1 (`d15e35e`, +7% interleaved).
- A2 step 1 subtract-elimination (`d4c9294`, −2.5% ins/byte).
- Bootstrap instrumentation (`ed4ca13`) — REFUTED Design B1 (post-flip clean = 2.7%).
- `inner_bench bootstrap` mode (`e68be90`) — MEASURED marker path = **27.4 ins/byte**
  (2.1× our clean loop, 3.1× libdeflate), on 31% of output. Mechanism: `deflate_block`
  runs a SEPARATE unoptimized loop (IsalLitLenCodePure + canonical distance + u16
  ring) instead of the FASTLOOP. **The collapse is the biggest quantified lever.**
- Reliable interleaved-delta + `inner_bench` harness built; `vector_huffman` deleted.

## Execution order — enablers first (each makes the next safer/easier)

**E1 — finish the dead-code delete (0b).** Shrinks the surface so the collapse
is legible and there's one fewer decoder to reason about. Dependency-ordered
(callers before modules), build+test green after each:
`double_literal` (pre-verified all-dead: `decode_huffman_double_lit`,
`decode_huffman_cf_double`, `get_fixed_double_lit_cache`, `bench_double_literal`)
→ bgzf's dead decode path (`decode_*_into`, `decompress_single_member_parallel`
+ tests) → then the now-orphaned LUT modules (`combined_lut`, `packed_lut`,
`simd_huffman`, `two_level_table`, `ultra_fast_inflate`).

**E2 — the golden fuzz/differential harness (0a).** THE safety net for the
collapse and everything after: a proptest + corpus differential that decodes
through the pure engine vs flate2 AND libdeflate, byte+CRC. Snapshot each
decoder's golden output vectors BEFORE it is deleted. Run in-process at the
decode-primitive level (NOT the parallel pipeline — avoids the known test
deadlock and isolates the engine).

**E3 — THE COLLAPSE (biggest lever).** Implement `Inflate<Markers>` u16
emission on the FASTLOOP so the marker decode shares the optimized loop
(27.4 → toward 13 + marker overhead on 31% of output). Markers mode = the
clean FASTLOOP + a per-match "distance > out_pos ⇒ emit marker instead of
copy" branch (rare); reuse the fast literal/length decode + bit refill.
Env-gated default-OFF; per-step byte-exact via E2 + `inner_bench bootstrap`;
interleaved-delta T8 wall to bank the win. Then route the bootstrap through it
and delete `deflate_block::Block`.

**E4 — A2 remainder.** Close the 1.20× resumable-contract tax + residual
algorithmic gap (pure 13 → libdeflate 8.6). Deterministic `inner_bench`.

**E5 — generalize (Phase 4).** Make Clean the sole path for single-member
T1/streaming, multi-member, BGZF, arm64/NEON — replacing ISA-L/libdeflate/zlib-ng.
Env-gated longest; flip default per-format only at ≥parity + byte-exact on the
full corpus. Highest ship-risk → gate longest.

**E6 — demolish (Phase 5).** Delete `isal-rs`/`libdeflate-sys`/`zlib-ng` from
Cargo (keep behind `dev`/`oracle`); update CLAUDE.md routing + release.yml.

**E7 — close/justify the remaining cells.** incompressible (stored-block bulk
memcpy fast path); bgzf-T1 (falls out of E5); tiny (static-link investigation —
flag as structural if irreducible); single-member T4-16 (the collapse + A2
move it; the T16-on-8-P-cores ceiling is structural — justify in writing).

## Honest caveats
- Multi-week effort. E3 and E5 are the long poles.
- Done-test line 5 includes 1-2 STRUCTURAL cells (process startup, physical
  core count) that a faster Huffman loop cannot close — the deliverable
  documents those with measured justification rather than pretending parity.
- Correctness is the prime directive: no unvalidated decoder change ships; every
  step is one gated, bisectable commit.
