# T1 IsalSingleShot routing change — disproof advisor verdict

Branch `owner/t1-singleshot-route`, uncommitted on base `d56cb0f5`.
Read-only review. Method: first-hand code trace (no build run — owner
self-reported 887/0 isal suite; x86_64 gzippy-isal can't build on this
arm64 host without Rosetta and the correctness questions are answerable
by reading the routing graph + the ISA-L kernel wrapper).

## Claim 1 — ROUTING GATING (the load-bearing risk): SOUND

`classify_gzip` (src/decompress/mod.rs:162) order is, top to bottom:
1. `has_bgzf_markers(data)` → `GzippyParallel` (mod.rs:163-165)
2. `is_likely_multi_member(data)` → `MultiMemberPar`/`MultiMemberSeq` (166-172)
3. SINGLE-MEMBER block, `#[cfg(parallel_sm)]` (179):
   - `#[cfg(isal_clean_tail)]` → `if num_threads <= 1 { return IsalSingleShot }` (195-200)
   - then `StoredParallel` / `ParallelSM`

So BGZF ("GZ" subfield) and detected multi-member are BOTH classified
and returned ABOVE the IsalSingleShot return — neither can reach it.
Verified first-hand. No stream that "must go parallel" can mis-route:
the only thing that reaches IsalSingleShot is a stream that already
fell through to the single-member tail AND `num_threads <= 1`.

- num_threads == 0: `<= 1` catches it. Correct and arguably safer
  (single-shot is a complete, valid full decode at any thread count;
  ParallelSM at 0 threads is the worse option). Library entry
  `decompress_bytes` does `num_threads.max(1)` anyway (mod.rs:258).

- The is_likely_multi_member 16 MiB scan-window limitation (format.rs:70,
  the historical large-first-member truncation case) is FULLY MITIGATED,
  with belt AND suspenders:
  * A multi-member stream whose 2nd member starts before 16 MiB IS
    detected → MultiMemberSeq at T1, never IsalSingleShot.
  * A multi-member stream that slips past the window reaches
    IsalSingleShot → `decompress_gzip_stream` ITSELF loops over trailing
    members: on `ISAL_BLOCK_FINISH` with `avail_in >= 2` and a `1f 8b`
    magic it re-inits ISA-L and continues (isal_decompress.rs:59-84).
    This is a NEWER, more-correct version than the single-shot deleted
    in 5e563dc — the loop is the explicit no-truncation fix. So even the
    mis-route decodes in full. No silent truncation path exists.

## Claim 2 — BYTE-EXACTNESS + NO-FALLBACK: SOUND

- CRC32+ISIZE: `state.crc_flag = IGZIP_GZIP` (isal_decompress.rs:34,76)
  makes ISA-L parse the gzip wrapper and verify CRC32 + ISIZE against the
  8-byte trailer. A mismatch (or any decode error / truncation) makes
  `isal_inflate` return non-zero → `return None` (47-49). Truncated input
  with no progress → `None` (85-87). `None` → `ok_or_else` → terminal
  `GzippyError::decompression` (mod.rs:516-518). NO fallback to ParallelSM
  or any other path — Rule 5 satisfied (contrast MultiMemberPar at
  mod.rs:301 which DOES fall back; IsalSingleShot deliberately does not).

- Input classes: empty input is short-circuited before classify
  (`data.len() < 2` → Ok(0), mod.rs:271-273) and inside the kernel
  (`input.is_empty()` → None, isal_decompress.rs:28). Stored blocks,
  fixed-Huffman, and a valid empty-payload gzip member are all handled
  by ISA-L's igzip kernel (the long-standing mature decoder). IsalSingleShot
  output must equal the true plaintext; ParallelSM (pure-Rust port) must
  too — ISA-L is itself the reference oracle, so if the two ever diverged
  ISA-L is the trustworthy side. Owner's dual-sha (both features, T1/T4/T8
  byte-exact) is consistent with this; no input class is left where
  IsalSingleShot could silently differ AND escape CRC/ISIZE detection.

## Claim 3 — T4/T8 NO-REGRESSION: SOUND

The route is guarded by an explicit `if num_threads <= 1` (mod.rs:197),
inside `#[cfg(isal_clean_tail)]`. It is structurally impossible to fire at
T>1. The added test asserts T∈{2,3,4} → ParallelSM on the isal build
(mod.rs:838-850). T4/T8 therefore stay ParallelSM byte-for-byte unchanged;
the 0.906 / 1.038 numbers describe the untouched path. (Aside, out of
scope for THIS change: T8 1.038x carries a reported 31% spread — that is a
pre-existing ParallelSM property, not introduced here; the central
estimate clears the 0.99 bar but the spread is worth noting for the
scorecard, not for banking this routing change.)

## Claim 4 — ONE PRODUCTION PATH: SOUND

The T1-vs-T>1 split is clean (single integer compare), documented in the
DecodePath doc comment (mod.rs:84-97) and the classify comment
(181-194), and ASSERTABLE via the new `EXPECT_PATH` knob threaded through
parity.sh / _parity_guest.sh (default `ParallelSM`; T1-isal cell sets
`IsalSingleShot`). It does not muddy "which function the CLI calls" — it
sharpens it: `GZIPPY_DEBUG=1` prints `path=IsalSingleShot` and the harness
fails closed if the observed path ≠ EXPECT_PATH. Re-introducing single-shot
is sound, not a stale resurrection: the function is the current, multi-member-
safe `decompress_gzip_stream`, the cfg coherence is airtight (see below),
and it is the only function the variant dispatches to. The DecodePath
enum variant is defined unconditionally (only `#[allow(dead_code)]`), so
all three dispatch match arms and the handler compile on every build;
only CONSTRUCTION is cfg-gated — no cfg/exhaustiveness hole. The pre-existing
`test_classify_single_member` (routing.rs:1462) calls classify at T=4, still
returns ParallelSM, still matches — unaffected.

### cfg coherence (the one way this could have been catastrophic): SOUND
`gzippy-isal = ["pure-rust-inflate", "isal-compression"]` (Cargo.toml:81)
⟹ has_gzippy_isal ⟹ CARGO_FEATURE_ISAL_COMPRESSION set. build.rs:110
`isal_clean_tail = is_x86_64 && has_gzippy_isal && parallel_sm`. Therefore
`isal_clean_tail` TRUE ⟹ (isal-compression ON) AND (x86_64), which is
exactly the cfg of the REAL `decompress_gzip_stream`
(isal_decompress.rs:24, `cfg(all(feature="isal-compression", target_arch="x86_64"))`).
The variant can NEVER be constructed on a build where the function resolves
to the `None`-returning stub (isal_decompress.rs:93). No "every T1 decode
Errs" failure mode. gzippy-native (isal_clean_tail false) never constructs
the variant → stays ParallelSM at every T.

## Claim 5 — SCOPE / SCORECARD: SOUND (no over-claim)

The change touches ONLY T1 single-member routing on gzippy-isal. Post-change
closable-cell scorecard vs the >=0.99x-every-T bar:
- T1: was ParallelSM ~0.905x (FAIL) → now IsalSingleShot 1.200x (PASS).
- T4: ParallelSM 0.906x (FAIL) — unchanged, pre-existing parallel-scheduling
  deficit, the sole remaining failing cell.
- T8: ParallelSM 1.038x (PASS, modulo the noted spread) — unchanged.
Accurate. The claim does not assert this fixes T4 and does not claim engine
credit — T1 is won by handing to the ISA-L igzip kernel, correctly framed.

## BOTTOM LINE: SAFE TO BANK / COMMIT

This is byte-exact and correctly gated. No mis-route hole: BGZF and
multi-member are classified strictly above the IsalSingleShot return, the
guard is `num_threads <= 1` (catches 0), and even a window-evading
multi-member stream decodes in full via the kernel's own member loop. CRC32
+ ISIZE are verified and a mismatch is a terminal Err with no fallback
(Rule 5). cfg coherence guarantees the real ISA-L kernel — never the stub.
T4/T8 are structurally untouched. No correctness FIX-NEEDED, none REFUTED.

Non-blocking note (not a gate): consider asserting `isal-compression` at
the IsalSingleShot dispatch arm with a `debug_assert!(cfg!(...))` or a
compile-time tie, purely as future-proofing if someone later loosens the
`isal_clean_tail` definition — today it is provably coherent.
