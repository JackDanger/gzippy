# DISPROOF-ADVISOR VERDICT — window-absent decodeBlock 1.6× attribution

Independent, read-only. Every claim source-verified first-hand at HEAD 7aae6c4a
(+ measurement overlay). Tried to break the owner's attribution; here is what survived.

## Bottom line
The **core attribution is UPHELD**: within gzippy's decodeBlock, the CLEAN u8 tail
(pure-Rust `resumable.rs` via `StreamingInflateWrapper`) is the dominant term, NOT the
u16 marker loop. That part is causally airtight. The **2.3×-vs-ISA-L magnitude and the
"actionable without more measurement" framing are NOT** — they ride on cross-tool
subtraction and on a decodeBlock SUM that the owner himself flags as slack-masked.

## Claim-by-claim

### UPHELD — the measured `isal_clean_tail` build's clean tail is pure-Rust, not ISA-L
Traced the production route end-to-end:
- `decode_chunk_unified_marker` → `MarkerStep::FlipToClean` handler (gzip_chunk.rs:949-973)
  calls `finish_decode_chunk_with_inexact_offset` (gzip_chunk.rs:965).
- That → `finish_decode_chunk_impl` (gzip_chunk.rs:511) → `StreamingInflateWrapper::with_until_bits`
  (gzip_chunk.rs:571) = `unified::Inflate<Clean,Generic,Streaming>` (inflate_wrapper.rs:154-161)
  = pure-Rust `resumable.rs`.
- The REAL-ISA-L path (`finish_decode_chunk_isal_oracle`, gzip_chunk.rs:160) is reached ONLY
  when `isal_engine_oracle_enabled()` (gzip_chunk.rs:539, env `GZIPPY_ISAL_ENGINE_ORACLE=1`)
  AND `feature="isal-compression"`. It is labeled "measurement-only, NOT a production path"
  (gzip_chunk.rs:149) and confirmed by gzip_chunk.rs:1635 ("the FlipToClean tail currently
  runs 100% through resumable").

**MOST LOAD-BEARING CORRECTION (a confirmation, not a refutation):** `build.rs:98-101`'s
comment — "route the chunk's clean tail through REAL ISA-L FFI instead of the pure-Rust
StreamingInflateWrapper" — is STALE/ASPIRATIONAL and directly contradicts the wiring. The
cfg name `isal_clean_tail` is a **misnomer in production**: on that build the clean tail is
pure-Rust. A reviewer who trusts the build.rs prose would wrongly REFUTE the owner; the code
proves the owner right. This strengthens the structural root cause and should be flagged so
the stale comment doesn't mislead the next reader.

### UPHELD — clean tail is the dominant decodeBlock term, NOT the u16 marker loop (angle A)
The two slow-knobs are cleanly const-generic-separated (marker_inflate.rs:1444-1465,
slow_knob.rs:79-114): `GZIPPY_SLOW_MODE` perturbs the `<false>` clean path (marker_inflate
seam drain + resumable.rs:1199); `GZIPPY_SLOW_MARKER_MODE` perturbs the `<true>` u16 careful
loop, each const-folding to 0 on the other specialization. So CLEAN+100% leaving the marker
body unchanged (312 ms) is the EXPECTED, correctly-isolated signature, and the +194/+248 ms
ΔdecodeBlock lands in clean decode. Minor caveat: `GZIPPY_SLOW_MODE` also fires in the
marker_inflate `<false>` arm, but on `isal_clean_tail` that arm only drains the one flip-seam
read before FlipToClean hands off — negligible vs resumable's ~139M bytes. Headline survives:
the delta is in **clean u8 decode, not the u16 marker loop**. The inject site (resumable.rs:1199)
is verified inside the post-flip clean engine.

### UPHELD — refutations (a) marker inner loop / (b) u16-over-bulk / (c) table-build
- (b) Flip threshold byte-identical to vendor: marker_inflate.rs:1116-1119
  (`distance_to_last_marker_byte >= MAX_WINDOW_SIZE && == decoded_bytes`) ↔ deflate.hpp:1282-1284.
  Post-flip the bulk routes to the u8 clean engine (gzip_chunk.rs:1397-1410), so gzippy does
  NOT run u16 over the clean bulk. UPHELD.
- (a)/(c) Consistent with the A/B: marker careful loop is the smaller decodeBlock term.

### UPHELD-WITH-CAVEAT — "marker loop is FASTER than rg" (0.68×) (angle B)
NOT verifiable from gzippy source. It compares gzippy's u16 marker body (0.323 s / 73.0M)
against rapidgzip's "custom inflate" verbose label (0.4748 s), whose denominator (window-absent
marker prefix ONLY vs prefix + pre-switch clean bytes + table-build) cannot be confirmed without
rapidgzip's verbose semantics. Denominator-mismatch risk is real. This is a **secondary, cross-tool**
claim — it is NOT what the causal A/B proves. Treat as suggestive, not established.

### UPHELD-WITH-CAVEAT — 2.3× per-byte clean ratio (angle C)
The owner already hedges this correctly (attribution.md:114-119): the ≈0.48 s clean-tail figure
is a SUBTRACTION (decodeBlock − marker body) and the 2.3× rests on rg's "ISA-L 0.2065 s" covering
≈the same ≈139M bytes — unverifiable here. The PROVEN claim is "clean tail is the bigger
decodeBlock term"; the 2.3× magnitude is a hypothesis. Endorsed as stated.

### REFUTED (as-framed) — "actionable without another measurement" (angle D)
This is the **second load-bearing correction**. The A/B measured **ΔdecodeBlock SUM**, not
Δwall, and the owner notes decodeBlock SUM is slack-masked at Fill 85% (attribution.md:46-47
caveat carried from the brief). Per Measurement PROCESS rule #3 (slow-down slope ≠ speed-up
ceiling; bound a speed-up by REMOVING the region, never by extrapolating the slow-down slope),
the +194/+248 ms slow-injection bounds nothing about how much closing the clean-tail gap moves
the WALL. The removal oracle ALREADY EXISTS and is wired (`GZIPPY_ISAL_ENGINE_ORACLE=1` →
`finish_decode_chunk_isal_oracle`, gzip_chunk.rs:160/539). It MUST be run to the WALL (not just
decodeBlock SUM, watching `ISAL_ENGINE_ORACLE_FALLBACKS==0` so the run isn't contaminated) before
a work-stretch. Attribution is sound enough to scope the fix **DIRECTION** (clean engine, not
marker loop); it is NOT sufficient to justify the work-stretch's payoff.

### CAVEAT — faithfulness of candidate 1 (angle E)
Candidate 1 (route post-flip clean tail through real ISA-L FFI) is faithful to rapidgzip's
WITH_ISAL build (charter goal #2) but conflicts with the governing pure-Rust north star
([[project_implementation_task_pure_rust_port]] "delete C-FFI from the decode graph" +
[[project_faithful_unified_decoder_over_perf]] "ONE u8-direct clean engine, no two-phase").
Both current candidates are still TWO-PHASE (Engine M → Engine C), so neither is the one-engine
u8-direct target. The owner's framing is accurate; just make explicit that candidate 1 re-introduces
C-FFI and is goal #2, while only candidate 2 advances goal #1.

## Verdict summary
| claim | verdict |
|---|---|
| measured clean tail = pure-Rust resumable, not ISA-L | UPHELD (build.rs:98-101 comment is stale — flag it) |
| clean tail dominates decodeBlock, NOT u16 marker loop | UPHELD (causal A/B, correctly isolated) |
| (a/b/c) refutations | UPHELD |
| marker loop 0.68× FASTER than rg | UPHELD-WITH-CAVEAT (cross-tool denominator unverified) |
| clean tail 2.3× slower per byte | UPHELD-WITH-CAVEAT (subtraction + cross-tool; hypothesis) |
| actionable without more measurement | REFUTED — run the existing ISA-L removal oracle TO THE WALL first |
| candidate 1 (ISA-L FFI) faithful | CAVEAT — faithful to rg-isal (goal #2), violates pure-Rust north star (goal #1) |

**Single most load-bearing correction:** The decodeBlock-internal attribution (clean tail >
marker loop) is causally proven and the structural read is correct (build.rs's "ISA-L"
comment notwithstanding — code wins). But a decodeBlock-SUM delta is NOT a wall verdict;
the speed-up ceiling is unbounded until the **already-wired `GZIPPY_ISAL_ENGINE_ORACLE` is
run to the WALL**. Do not scope a clean-engine work-stretch on the slow-down slope alone.
