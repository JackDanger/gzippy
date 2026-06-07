# DISPROOF-ADVISOR VERDICT — bounded pure-Rust engine ceiling (VAR_VI)

**VERDICT: PLATEAU UPHELD-WITH-CAVEATS**

The *engine-isolation* claim (a faithful pure-Rust+ASM DEFLATE engine reaches only
≈0.6× ISA-L, well short of the 0.85 igzip-class bar) is **UPHELD** and survives an
adversarial source review — it is robust even after crediting every measurement
caveat I found. The brief's *downstream* claim ("the production **1.0× WALL** is
HARD-BOUNDED at ~0.6×") **OVERREACHES** what isolation can prove and is the load-
bearing caveat below.

Read in full: `plans/engine-ceiling-advisor-brief.md`. Source-verified read-only:
`benches/engine_isolation.rs`, `src/decompress/parallel/lut_huffman.rs`,
`src/backends/isal_decompress.rs:307-419`.

---

## Q1 — Is VAR_VI a faithful "best pure-Rust+ASM", or is a technique missing/crippled?

Every claimed technique is genuinely LIVE on a `target-cpu=native` (BMI2+AVX2) build:

- **BMI2 BZHI** — `bzhi64` → `core::arch::x86_64::_bzhi_u64` (`:395-397`), called for the
  variable-width distance extra-bits at `:875` (fast) and `:955` (tail). REAL. Scope is
  honestly narrow: only distance-extra extraction; litlen masking is AND-imm already. Its
  measured contribution is correctly small.
- **AVX2/SSE overlap copy** — `avx_backref_copy` uses real `_mm256_loadu/storeu_si256` and
  `_mm_loadu/storeu_si128` (`:409-453`), called at `:885`/`:976`. REAL.
- **Speculative 8B store** — `(out_ptr.add(out_pos) as *mut u64).write_unaligned(packed)`
  at `:826-829`, `packed = sym & 0x00FF_FFFF`; out_pos advances only by the *actual* literal
  count (`lit_prefix`, `:830-849`). Slop-licensed, correct, and exactly the igzip multi-literal
  write. REAL.
- **Packed-u32 multi-symbol short table (trick #3)** — confirmed in `lut_huffman.rs:1021-1063`:
  `decode` returns `sym_count` (≤3) and VAR_VI unpacks up to 3 packed literals per decode.
  Fully exploited.
- **PRELOAD pipeline** — `:812` (preload) and `:892` (preload-next). REAL.

**Tail does not dominate / header-build is symmetric.** The fast loop decodes each block to
its EOB (`trailing_code==END_OF_BLOCK_SYMBOL ⇒ at_eob=true; break 'fast`, `:852-854`), after
which `if !at_eob` (`:899`) SKIPS the careful tail. The bounds-checked tail fires only when
`out_ok/in_ok` force a mid-block break near the 4 MiB target or input exhaustion — i.e. once
per run, not per block. So the measured path IS the speculative fast loop. `build_block_tables`
(`:805`) is inside the timed region but so is ISA-L's own header parse (it is handed the same
`data, start_bit, dict, n_actual` and parses headers internally, `isal_decompress.rs:315-362`),
so the denominator is symmetric on header/table cost.

**Two real but small under-representations of "best ASM" (both make VAR_VI look slightly
SLOWER than achievable — they cut AGAINST plateau, and still don't close the gap):**

1. **Small-overlap copy is scalar, not igzip doubling.** For `2 ≤ distance < 16` overlapping
   back-refs, `avx_backref_copy` falls to a byte loop (`:447-452`) instead of the
   write-`distance`-then-double-with-16B technique. Short distances are common in text/binary,
   so this is a genuine deviation from the vendor copy. Magnitude unknown but bounded — it
   cannot plausibly be ~23pp.
2. **`consume(bit_count)` is a plain shift, not an explicitly-emitted SHRX** in the bit reader;
   the brief asserts SHRX but it is left to the compiler. With `native` this often lowers to
   SHRX anyway, so not clearly crippled.

Crediting both, plus the Q4 copy asymmetry, lifts a plausible "fixed" VAR_VI from ~0.60 to at
most ~0.65–0.68 — **still ~17–20pp under the 0.85 PASS line**, and the result is stable across
two independent runs (0.59–0.62). The deeper reason cuts FOR plateau: igzip's edge is a fully
hand-scheduled asm hot loop; a Rust port routed through a `DecodedSymbol` struct return + a
`while remaining` unpack carries branch/codegen overhead that is exactly the thing the plateau
asserts pure Rust cannot erase. No missing technique large enough to reach 0.85 was found.

## Q2 — Is the byte-exactness reasoning airtight?

**Yes, airtight, for the measured `[0, n_actual)` region.** `:1143` prints an MBps line iff
`exact[k]` (else "VOID"). `exact[k] = o.len() >= n_actual && &o[..n_actual] == scalar &&
scalar_eq_isal` (`:1085`), with `scalar = outs[0][..n_actual]` (VAR_I) and `scalar_eq_isal`
requiring `scalar == outs[2][..n_actual]` (ISA-L). These are real `==` slice byte-compares, not
CRC/hash, so no collision risk. Therefore an MBps line for VAR_VI ⇒ VAR_VI == VAR_I == ISA-L
byte-for-byte over the entire timed window. Two correct framings in the brief verified: (a) only
`[0, n_actual)` is validated, but that is exactly `target_n` fed to every variant, so the timed
region is fully covered; (b) the top-line `SHA_ALL_EQUAL=no` is `all_equal` (`:1087`, AND over
all 9 incl. the untouched VAR_IV_E234 failures) and is independent of `exact[8]` — it does NOT
impugn VAR_VI. (Caveat: I verified the print⇒exact LOGIC, not the asserted run output; that the
runs printed MBps on every chunk is the brief's measurement, taken on trust per its protocol.)

## Q3 — Is 0.85 the right bar, or could a 0.6× isolation engine still TIE the 1.0× WALL?

**This is where the brief overreaches, and it is the load-bearing caveat.** Isolation measures
single-thread, clean-chunk pure-decode throughput. The production *parity goal* is a PARALLEL
WALL (T8/T16) in which the engine is ONE stage among block-finding, marker resolution, window
application, and consumer/scheduling. The project's own surviving findings record that the wall
is NOT purely engine-bound at all T: `MEMORY` — "PLACEMENT + ENGINE are CO-PRIMARY", and the
T8 binder was LOCATED to serial/consumer-wait. The Measurement PROCESS rule #3 is explicit:
"slow-down slope ≠ speed-up ceiling; to BOUND a speed-up lever you must REMOVE the region and
measure — never extrapolate." The leap from "engine ≈0.6× ISA-L in isolation" to "the 1.0× WALL
is HARD-BOUNDED at 0.6×" is precisely the forbidden extrapolation through an unlocated knee.

- What isolation DOES license: a pure-Rust engine cannot be made *igzip-class as a standalone
  primitive* on this design. UPHELD.
- What it does NOT license: that the production parallel wall is therefore stuck at 0.6× of
  rapidgzip. If at the target T the wall is gated by a shared serial/consumer stage (as the
  campaign already located), a ~0.6× per-thread engine can still be off the critical path and
  the wall can TIE — OR the 2.3× clean-rate engine gap survives placement (also on record) and
  it cannot. Isolation cannot adjudicate this; only a parallel-wall causal perturbation can.

So the 0.85 bar is the right bar for the *engine-primitive* question it gates, but it is the
WRONG instrument for the *1.0×-vs-no-FFI fork* the verdict invokes. The fork is REAL; its
hard-bound is NOT established at 0.6× by this bench.

## Q4 — Confounds in the ratio

- **Asymmetric final copy (real, small, against VAR_VI).** VAR_VI ends with
  `out[base..end].to_vec()` (`:993`) — a full ~n_actual (≤4 MiB) memcpy every iter. ISA-L ends
  with `output.truncate(out_pos)` (`isal_decompress.rs:417`) — O(1), no copy. On an ~8 ms VAR_VI
  decode a ~4 MiB copy is ~0.2–0.4 ms ≈ a few percent, charged to VAR_VI only. Inflates the gap
  by a few pp; removing it would help VAR_VI, not refute the verdict.
- **Zero-alloc is ~symmetric.** Both allocate `vec![0u8; ~n_actual]` per call (`:759` vs
  `isal_decompress.rs:372`); VAR_VI's cap is marginally larger (window prefix + slop).
- **32 KB window prime ~symmetric.** VAR_VI `copy_from_slice` of 32 KB (`:760`) vs ISA-L
  `isal_inflate_set_dict` of 32 KB (`:351-362`).
- **Environment (load ~3.3, turbo on, core-0 pin).** All variants are interleaved per-iter
  within one process (`:1120-1127`) and the verdict uses RATIOS, so frequency/turbo/load drift
  is first-order common-mode and cancels. SELFTEST iii/i = 2.73 ∈ [2.5,3.6] is a sane sanity
  gate. `black_box(&r)` (`:1125`) prevents dead-code elision. No ratio-distorting confound found
  beyond the final-copy asymmetry, which is small and points the safe direction.

---

## Load-bearing reasons

1. **UPHELD (engine primitive):** All five igzip techniques are verifiably live; the fast loop
   (not the tail) is the timed path; the denominator is header-symmetric; byte-exactness vs both
   scalar and ISA-L is airtight. The ~0.60× result is stable across two runs and the only
   asymmetric confound (final to_vec) plus the two minor crippling caveats together cannot lift
   it past ~0.65–0.68 — ~17–20pp short of 0.85. Pure-Rust igzip-class as a standalone engine is
   not reached here, and the structural reason (Rust-vs-hand-asm hot-loop codegen) supports the
   plateau rather than promising a hidden lever.

2. **CAVEAT (the overreach):** The verdict's escalation to "the 1.0× WALL is HARD-BOUNDED at
   ~0.6×" is NOT supported by isolation and contradicts the campaign's own located finding that
   the parallel wall is co-gated by placement/consumer-wait. Per Measurement PROCESS #3, bounding
   the wall requires REMOVING the engine stage in the production parallel pipeline and measuring —
   not extrapolating the isolation ratio. Treat the engine ceiling as established; treat the wall
   bound as an OPEN question owed a parallel-wall causal perturbation before any "no-FFI fork is
   hard-bounded" decision is made on it.

3. **Minor, non-blocking:** small-overlap (`2 ≤ dist < 16`) copy is scalar rather than igzip
   doubling, and SHRX is compiler-discretionary. Worth closing before any *final* ceiling claim,
   but neither changes the verdict.
