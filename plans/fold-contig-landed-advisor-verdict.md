# DISPROOF VERDICT — LANDED copy-free FOLD clean-drain + ContigFoldSink default

Independent, read-only. Source-verified first-hand against the working tree on branch
reimplement-isa-l (HEAD 7aae6c4a + uncommitted overlay). Compile-checked the production
path (`cargo check --no-default-features --features pure-rust-inflate
--target x86_64-apple-darwin` → **exit 0**, clean). I tried to break L1/L2/L3.

## Summary up front
- **L1 mechanism (copy-free drain + contig sink): CONFIRMED.** The owed control I asked
  for in the prior verdict was actually run and landed: `drain_to_output`'s post-flip
  clean branch is now copy-free. The prior verdict's prediction held — removing copy #1
  MOVED the wall (+0.040×), so a free cadence component HAD been mis-booked as intrinsic.
- **L1 magnitude: OVERSTATED.** The honest banked number is **+0.059×** (0.678→0.737,
  quiet, default binary). **+0.083×** is the LOADED 6-pass split-sum and should not be the
  headline. The source comments (gzip_chunk.rs:841, 948–951) hard-code +0.083× — fix them.
- **L2 (residual ~0.20× = intrinsic symbol rate): STILL OVER-ATTRIBUTED.** Smaller flaw
  than last time, same shape: the oracle ceiling `ocl_cf` is ring-free AND copy-free AND a
  different engine (ISA-L), while copy-free FOLD still pays the ring-write + the
  ring→`chunk.data` drain memcpy. The 0.188× remainder is not cleanly "symbol rate."
- **L3 (ContigFoldSink default, delete UnifiedMarkerSink): CONFIRMED.** No residual refs,
  no env gate, compiles clean, two-phase `CleanTailSink` path untouched, sink overrides
  correct, and decode correctness is independent of the sink anyway (it's an accumulator).
- **Reserve safety: SAFE, but the "worst DEFLATE ratio" justification is FALSE.**

---

## L1 — mechanism CONFIRMED, magnitude OVERSTATED

**Source (marker_inflate.rs:745–784).** The `!contains_marker_bytes` branch now pushes the
≤2 CONTIGUOUS u8 ring slices straight to the sink via `push_clean_u8(from_raw_parts(...))`
— no `Vec::with_capacity(new_bytes)`, no byte-by-byte fill. This is exactly the control
the prior verdict said was owed (the symmetric of the marker branch's `push_slice`).

**Byte/accounting exactness — checked the wrap edge.** Old path stitched the physical wrap
into ONE linear `u8buf` (`(ring_drained+i) % U8_RING_SIZE` per byte) then one
`push_clean_u8`. New path emits TWO `push_clean_u8` calls across the wrap. For
`ContigFoldSink::push_clean_u8` (gzip_chunk.rs:860–877) that is: incremental CRC (assoc.
across slices ✓), two `extend_from_slice` ✓, summed `non_marker_count` / `decoded_size` /
`clean_appended` ✓. Same bytes, same order, identical accounting even on the wrap split.

**Magnitude is inconsistent with itself.** L1 says "+0.083× … native_fold 0.678→0.737×."
0.737 − 0.678 = **+0.059×**, not +0.083×. The +0.083× is the sum of two sub-component
deltas (copy#1 +0.040 + copy#2/3/grow +0.043) measured in the 6-pass split under **load
1.4–2.8**; the +0.059× is the quiet 3-pass default-binary banked delta. Under load, copy
costs inflate (less turbo headroom, more memory-bus contention), so the split-sum is the
load-inflated figure. **The honest banked recovery is +0.059×.** The 6-pass split is fine
as evidence of SIGN and MONOTONICITY (old<new_off<new_contig every pass — that's a real,
robust ordering), but its magnitude is not bankable. The source comments repeating "+0.083×
of the T8 wall" overstate by the load factor and should be corrected to the banked number.

Net: keep the change (correct, byte-exact, removes a real per-block alloc + byte loop;
rule 7a). Headline it as **+0.059× banked**, with the split as sign/ordering support only.

## L2 — residual still over-attributed (disproof angle 4 confirmed)

The split DID shrink the "intrinsic" remainder: 0.678→0.925 (0.247×) became 0.737→0.925
(**0.188×**), and that's the right method (oracle-removed ceiling, not VAR_VI slope; Rule
3). But "the residual ~0.20× is intrinsic symbol rate" still doesn't hold cleanly, for the
reason disproof angle 4 names:

`ocl_cf` decodes ISA-L **directly into `writable_tail_reserve`** (segmented_buffer.rs:206–
217) — one contiguous FFI write into `chunk.data`, **no ring, no drain**. Copy-free FOLD,
even after this change, still pays:
1. the engine **ring write** (literal/backref store into `output_ring`), then
2. the **ring → `chunk.data` drain memcpy** (`extend_from_slice` in `push_clean_u8`).

"Copy-free" here means *no `u8buf` middle-man* — but the ring is not `chunk.data`, so the
drain `extend_from_slice` is still a second touch of every clean byte that `ocl_cf` does
not pay. So the 0.188× confounds THREE differences at once: symbol rate (pure-Rust engine
vs ISA-L) **+** the ring-write **+** the ring→data drain copy. L2 books the whole thing as
symbol rate. A clean split would need a same-engine (pure-Rust), ring-based oracle that
writes its final bytes copy-free — which does not exist. So L2 is **directionally right and
better-bounded than before, but NOT licensed to call 0.188× "intrinsic symbol rate."** The
true intrinsic-rate gap is ≤ 0.188×; some of it is still the ring-drain memcpy + ring write.

## L3 — CONFIRMED (default + deletion correct, blast radius bounded)

- **Deletion real.** `struct UnifiedMarkerSink` was added in f3e383eb and is GONE from the
  working tree; grep finds zero references outside historical comments. The
  `GZIPPY_FOLD_CONTIG` env gate is gone — `ContigFoldSink` is unconditional
  (gzip_chunk.rs:960, the sole sink in `decode_chunk_unified_marker`). The only env vars
  left in the file are `GZIPPY_ISAL_ENGINE_ORACLE` and `GZIPPY_MARKER_RING`.
- **Compiles clean** (exit 0) — the deletion left no dangling references.
- **Two-phase path untouched.** `CleanTailSink` (gzip_chunk.rs:1037–1082) is a distinct
  struct routing through `append_clean` / `append_clean_narrowed`; it is the
  `isal_clean_tail` Design-A sink and this change does not touch it. ✓
- **Sink overrides verified for the pre-flip window path (disproof angle 2):**
  - `trailing_clean_since` (887–898): I traced the marker++clean logical layout. For
    `from ≥ marker_len` it returns `clean_len − (from − marker_len)` = the clean run (all
    post-flip bytes are clean). For `from < marker_len` it returns
    `markers.trailing_clean_since(from) + clean_len` — correct because the markers slice's
    trailing clean run ends at `marker_len`, which is *physically adjacent* to the
    contiguous clean region, so the two runs concatenate. Matches `block_len` exactly when
    a block is wholly clean (the gate at 1517). ✓
  - `is_last_n_clean` is NOT overridden → uses the trait default (98–104) which routes
    through the overridden `trailing_clean_since` → correct.
  - `copy_last_n_clean_u8` (899–907) returns `false` when `clean_appended ≥ n`. This looks
    degenerate but is harmless: it has **no live external caller** (grep: only the
    self-delegation at :904), and the native window is taken from the engine ring /
    `last_32kib_window_vec`, never from this method. The `last_32kib_window_vec` call site
    (:992) is the dead `FlipToClean` arm (native `marker_decode_step` never returns
    `FlipToClean` — confirmed last verdict via the `isal_clean_tail` cfg + live
    `flip_to_clean=0`).
- **Blast radius is bounded by design.** The sink is an OUTPUT ACCUMULATOR. Decode
  correctness lives in the engine ring (`flip_repack_to_u8`, back-refs resolve from
  `output_ring`), independent of the sink's predicates. So the worst a sink-override bug
  could do is perturb subchunk boundaries / window-detection accounting — never the decoded
  bytes. The sha match + 857 passing tests + flip-seam differential-vs-flate2 tests
  (gzip_chunk.rs:1833, 1874) cover that. L3 is sound.

## Disproof angles, answered

1. **+0.083× real or artifact? Reconcile with +0.059×.** Neither is "wrong," but they
   measure different things. +0.083× = loaded 6-pass split-sum (cost-inflated, but
   sign/monotonicity robust across all 6 passes — that ordering is the real result).
   +0.059× = quiet default-binary banked. **The honest banked number is +0.059×.** The
   code comments should not carry +0.083× as the headline.
2. **Did deleting UnifiedMarkerSink break anything subtle?** No — verified above. Overrides
   correct, no live caller of the degenerate `copy_last_n_clean_u8` branch, decode
   correctness independent of the sink, two-phase path untouched, compiles clean.
3. **Is the compressed×8 reserve safe?** SAFE — but the justification is false.
   `reserve_clean → SegmentedU8::reserve → Vec::reserve` is lazy capacity (no memset);
   `extend_from_slice` regrows on under-reserve (amortized doubling). So **under-reserve =
   safe regrow, never corruption.** HOWEVER the comment "compressed × 8 covers the worst
   DEFLATE ratio" is **wrong**: DEFLATE's max expansion is ~1032:1, not 8:1. For a
   highly-compressible chunk (e.g. long zero runs) ×8 UNDER-reserves and the buffer
   regrows — safe, but it silently defeats the "no regrow / fully copy-free" claim for
   those chunks. For silesia (~2–3:1) ×8 comfortably over-reserves, so the banked
   measurement is fine. Also note the over-reserve side: ×8 × concurrent chunks is a real
   (bounded) RSS bump at high T × large chunks — not OOM on a normal box, but worth a cap.
   Recommend: fix the comment, and consider clamping the reserve (e.g. to the chunk's
   decoded-size hint) rather than a blanket ×8.
4. **Is L2's 0.20× attribution sound?** No (see L2). `ocl_cf` still pays strictly less than
   copy-free FOLD (ring-free + copy-free-to-final + different engine), so the 0.188×
   remainder is not purely symbol rate.

## Required corrections before L1/L2 stand as written
1. Re-label the banked recovery **+0.059×** (not +0.083×) in L1 and in the source comments
   at gzip_chunk.rs:841 and 948–951. Keep the 6-pass split only as sign/monotonicity
   evidence.
2. Down-scope L2: "residual ≤ 0.188×, an UPPER BOUND on intrinsic symbol rate that still
   includes the ring-write + ring→data drain memcpy that `ocl_cf` does not pay." Do not
   call it "intrinsic symbol rate" until a same-engine ring-based copy-free-to-final oracle
   isolates it.
3. Fix the reserve comment (×8 ≠ worst DEFLATE ratio) and consider clamping the reserve.

## Bottom line
- L1 mechanism (copy-free drain + contig default): **CONFIRMED**, byte-exact, correctly
  banked as production. Magnitude **OVERSTATED** — honest banked is **+0.059×**, not
  +0.083×.
- L2 (residual = intrinsic symbol rate): **NOT LICENSED** — better-bounded (0.247→0.188×)
  and right method, but `ocl_cf` confounds ring + drain-copy + engine, so 0.188× is an
  UPPER BOUND on symbol rate, not symbol rate.
- L3 (ContigFoldSink default, delete UnifiedMarkerSink): **CONFIRMED** — deletion clean,
  compiles, two-phase path unaffected, overrides correct, blast radius bounded.
- Keep the change. Fix the three label/comment corrections above.

=== ADVISOR EXIT 0 ===
