# ENGINE BENCH ROUND 2 — E2-E4 implementation spec (for the impl subagent)

You are implementing E2-E4 inner-Huffman techniques in the STANDALONE engine-isolation
bench. This is a BENCH prototype to settle whether pure-Rust+inline-ASM can approach
igzip-class clean decode rate. NOT production integration (separately gated later).

Repo: /Users/jackdanger/www/gzippy-reimplement-isal, branch reimplement-isa-l, HEAD 5d5fc3b9.
Build LOCALLY via Rosetta (x86_64) for byte-exact iteration ONLY — do NOT run perf numbers
locally (Rosetta MB/s are illegitimate; the leader runs the locked guest for numbers). Your
job: make it COMPILE and pass BYTE-EXACT on x86_64, then report.

## SOURCE FACTS (already verified by the leader — do not re-derive, just build on)
- Production clean dynamic-block loop = `read_internal_compressed_specialized::<false>`
  (src/decompress/parallel/marker_inflate.rs:1191). This is what bench variant (i)/(ii) drive.
- Back-ref copy = `emit_backref_ring::<false>` (marker_inflate.rs:2137) — writes into a
  `output_ring: Box<[u16; RING_SIZE]>`. Already does 8-byte-word (4-u16) copies + RLE fill.
- Literal store = single u16 write (`:1348`). TRIPLE_SYM unpacks 1-3 packed literals one u16
  at a time (`:1343` loop).
- Refill = `bits.refill()` once per outer iter (`:1324`), libdeflate-style 56-63 bits.
- Clean-mode bytes are GUARANTEED < 256 (the `<false>` const generic; markers never appear).
  THIS is why a u8 ring is byte-exact in clean mode — the prime traffic lever (u16 ring =
  2 bytes/byte mem traffic vs ISA-L's u8 buffer).
- Primitives `lut_litlen_decode`, `dist_hc`, `output_ring`, `ring_pos`, `decoded_bytes` are
  PRIVATE on `Block`. `set_initial_window`, `read_header`, `eob`, `read`, `is_last_block`,
  `contains_marker_bytes` are pub. DISTANCE_BASE/DISTANCE_EXTRA, END_OF_BLOCK_SYMBOL,
  RING_SIZE, MAX_WINDOW_SIZE, MAX_RUN_LENGTH constants are module-level.

## THE APPROACH — new pub clean-only entry on Block (exercises REAL LUT primitives)
Add to `impl Block` in marker_inflate.rs a new method, cfg-gated identically to
`read_internal_compressed_specialized` (the `pure_inflate_decode`/isal x86_64 cfg):

    pub fn read_clean_e234(&mut self, bits: &mut Bits, n_max_to_decode: usize)
        -> Result<usize, BlockError>

It is a clean-ONLY sibling of `read_internal_compressed_specialized::<false>` (copy that
function as the starting point, drop the CONTAINS_MARKERS generic = always false, delete all
`if CONTAINS_MARKERS` dead branches). Then layer the techniques. It MUST require
`!self.contains_marker_bytes` (debug_assert) and decode byte-identically to the `<false>` path.

Wire a public driver `pub fn read_clean_e234` is reachable; add a sibling to `read` is NOT
needed — the bench calls `read_header` + a loop calling `read_clean_e234` directly + a drain.
You need the bench to be able to drain: add a `pub fn drain_clean_u8(&mut self, out: &mut Vec<u8>)`
that mirrors `drain_to_output`'s clean branch (narrow ring slot u16->u8 OR, after E1, copy u8
directly) so the bench can collect output. Keep it simple and correct.

### E1-full (u8 ring view) — FOUNDATION, do FIRST
The ring stays `Box<[u16; RING_SIZE]>` (faithful; markered prefix needs u16). But in CLEAN
mode reinterpret the ring backing as u8 for the clean region, vendor deflate.hpp:806,1742-1785
(`reinterpret_cast` u8 view). SIMPLEST byte-exact prototype: keep writing u16 BUT measure E2/E3/E4
on the u16 ring first (lower risk), THEN attempt a u8-reinterpret variant. Two sub-variants are
fine. PRIORITIZE getting E2/E3/E4 measured on the u16 ring (byte-exact, lower risk); the u8-ring
reinterpret is the stretch goal — if it threatens byte-exactness, report it as not-done and keep
the u16 version. NEVER ship non-byte-exact.

### E2 — wide SIMD back-ref copy
In `emit_backref_ring`'s non-overlap arm, current code does 8-byte (4-u16) word copies. Widen
to AVX2 32-byte (16-u16) copies for length >= 8 u16, falling back to the existing word loop for
the tail. Use std::arch::x86_64 `_mm256_loadu_si256`/`_mm256_storeu_si256` under
`is_x86_feature_detected!("avx2")` OR target-cpu=native (the build uses -C target-cpu=native so
AVX2 intrinsics are available without runtime detect — but guard with cfg(target_feature="avx2")
or is_x86_feature_detected to be safe). Respect the ring-wrap + the dist>=length non-overlap
invariant EXACTLY (hazard 05a3835: rounded word-copy corrupts repetitive data via ring overshoot
— keep the `*_round_fits` guards, only widen the stride). For the u8-ring variant, the SIMD copy
moves 32 BYTES = 2x logical bytes per op vs u16 (the real win).

### E3 — packed multi-literal store
At marker_inflate.rs:1343, the TRIPLE_SYM loop writes 1-3 literals one u16 at a time. Collapse:
when sym_count is 2 or 3, build the packed value and do ONE wide store (u16: a u32/u64 write of
2-3 u16 lanes after masking each to its byte; u8: a 2-3 byte store). ca52389 regressed this on
the OLD loop — NON-BINDING, re-measure. Keep byte-exact: each lane is `(sym >> (8*k)) & 0xFF`.

### E4 — wide refill amortized over multiple symbols
Currently refill() once per outer iter. After a refill, bitsleft in [56,63]. A literal-only
TRIPLE_SYM consumes ~12-36 bits; a back-ref ~48. E4: when the next 1-3 symbols are known to fit
(headroom check), decode multiple symbols between refills (elide the per-iter refill when
bitsleft already >= worst-case-for-the-packet). Mirror the FASTLOOP-margin idea
(read_internal_compressed_canonical_specialized:1674-1759 has refill_fast! as a reference). Keep
the SAFE tail for the last 32 input bytes / EOB.

## BENCH WIRING (benches/engine_isolation.rs)
1. Add variant (iv): `decode_var_iv_e234` — clean-primed Block, loop `read_header` +
   `read_clean_e234` until last/target, collect via `drain_clean_u8`. Mirror decode_var_ii.
   If you build incremental sub-variants (e2 only, e2+e3, e2+e3+e4 stacked), add them as
   separate fns and print each — the charter wants per-technique AND stacked numbers.
2. CHUNK SWEEP: change `run()` to sweep >=3-5 seed entries (e.g. indices at 10%, 30%, 50%,
   70%, 90% of the sorted seed list, each requiring a 32 KiB window — skip entries whose
   window != MAX_WINDOW_SIZE). For EACH chunk: byte-exact gate (all variants == scalar AND
   == ISA-L over n_actual) + per-variant median MB/s. Print per-chunk lines AND an aggregate
   (median of per-chunk medians, and min/max spread). Keep the existing self-test on the
   median chunk.
3. Byte-exact gate stays ABSOLUTE: every variant (i, ii, iv-stack) full `==` memcmp vs scalar
   AND vs ISA-L. A mismatch VOIDS that variant's number (print VOID, do not crash the sweep).
4. Print labels: VAR_IV_E2, VAR_IV_E23, VAR_IV_E234 (whichever you implement), with MBps_med
   and ratio vs (i) and vs (iii=ISA-L).

## CRITICAL BUILD NOTE — Rosetta is x86-64-v2 (NO AVX2)
Local Rosetta builds use `RUSTFLAGS="-C target-cpu=x86-64-v2"` (SSE4.2, NO AVX2). The guest
uses `-C target-cpu=native` (full AVX2). So your AVX2 path MUST be runtime-gated with
`is_x86_feature_detected!("avx2")` (NOT a compile-time cfg(target_feature)) so it compiles on
BOTH and the scalar/word fallback runs under Rosetta. Byte-exactness is verified via the
fallback locally; the AVX2 path's speed is measured on the guest. The AVX2 and fallback paths
MUST produce identical bytes (test both: you can force the fallback by also exercising the
existing emit_backref_ring path which the byte-exact gate compares against). Local build cmd:

    RUSTFLAGS="-C target-cpu=x86-64-v2" bash scripts/cargo-lock.sh \
      cargo build --release --target x86_64-apple-darwin \
      --no-default-features --features pure-rust-inflate,isal-compression --bench engine_isolation

Then run the produced bench bin under Rosetta (it runs natively via Rosetta on arm64 Mac).
NOTE: the bench needs /tmp/engine.seed (GZSEEDW2). Generate it locally by running gzippy with
GZIPPY_FORCE_PARALLEL_SM=1 GZIPPY_SEED_WINDOWS_CAPTURE=/tmp/engine.seed on the silesia corpus
at benchmark_data/silesia-gzip.tar.gz (same as guest_engine_isolation.sh:64-67) — build the
gzippy bin the same Rosetta way. If the corpus or seed capture is not feasible locally, report
that as a blocker and at minimum prove it COMPILES + the scalar paths are unchanged.

## GATES (you MUST pass before reporting DONE)
- Builds: `bash scripts/cargo-lock.sh cargo build --release --no-default-features --features
  pure-rust-inflate,isal-compression --bench engine_isolation` on x86_64 (Rosetta:
  `arch -x86_64 ...` or the existing rosetta toolchain — check reference_local_x86_tests_rosetta).
  SERIALIZE via scripts/cargo-lock.sh. Run `df -h .` before building.
- Byte-exact: run the bench locally under Rosetta; SHA_ALL_EQUAL must be yes for every chunk and
  every implemented variant. (MB/s under Rosetta are GARBAGE — ignore them; only correctness.)
- `cargo fmt` the touched files (pre-commit hook enforces it).
- Keep marker-mode + the existing <true>/<false> paths UNTOUCHED and still compiling.
- Do NOT wire read_clean_e234 into production `read()` routing — it is bench-only.

## REPORT BACK (to stdout, the leader reads your final message)
- Which techniques you implemented (E2 / E3 / E4 / E1-full-u8-ring), and which (if any) you
  could NOT make byte-exact and why.
- Confirmation the bench builds on x86_64 and SHA_ALL_EQUAL=yes for all swept chunks under
  Rosetta (paste the relevant bench output lines).
- The exact new fn names + file:line so the leader can review.
- Anything that blocks the guest run.
Do NOT commit — the leader commits after review. Do NOT run guest/neurotic anything.
