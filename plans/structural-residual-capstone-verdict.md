# STRUCTURAL-RESIDUAL CAPSTONE — disproof gate (Opus advisor, read-only, 2026-06-09)

Gate on the low-T BAR-1 structural-residual sizing (bin 2d317027, isal_chunks=14@T4/16@T1,
frozen guest, interleaved N=13-15, /dev/shm, sha-verified). Source-verified the two oracle
mechanisms first-hand (`chunk_fetcher.rs:3884-3947`, `marker_inflate.rs:1490-1517`).
Reconciled against `pipeline-fidelity-verdict.md` + the ISA-L glue audit. ATTACK, not ratify.

---

## CLAIM 1 — OUTPUT = proved SHARED floor.  VERDICT: SOUND.

The dual-sided removal is sound. Source-confirmed (`chunk_fetcher.rs:3896-3914`): the
`GZIPPY_SKIP_WRITEV_SYSCALL` branch still BUILDS the iovecs (`:3896-3903`), still combines
CRC (`:3899-3902`), still `*total_size +=` and still `defer_chunk_recycle` — it drops ONLY the
`writev` syscall. So the oracle removes output and nothing else; no over-removal inflates the
delta.

The SKIP_WRITEV-vs-/dev/null asymmetry the owner flagged is REAL but cannot manufacture the
conclusion: gz removes the whole syscall (no entry, no copy), rg-null removes the 211 MB
copy_from_user but still pays write() entry + iovec walk. That asymmetry (i) is sub-ms in
magnitude (16 syscall entries) and (ii) points the FAVORABLE way for gz — so gz's true output
cost is >= the measured 86ms, which only STRENGTHENS "shared floor." The load-bearing proof is
not the 86 ≈ 85 near-equality alone but the ratio-INVARIANCE under dual removal (T1
0.878->0.867, T4 0.861->0.862): removing output from BOTH leaves the gz-vs-rg ratio unchanged,
i.e. output is not a differential term. rg's matched serial writeFunctor
(`ParallelGzipReader.hpp:521`) is the structural existence proof. NOT a lever. SOUND.

---

## CLAIM 2 — MARKER-BOOTSTRAP = on critical path but SHARED, ~0 net excess.
## VERDICT: FIX-NEEDED (one reasoning step); conclusion stands on the OTHER argument.

Two facts are clean: the perturbation (spin +50%->+68ms/+100%->+155ms, monotonic; sleep
freq-neutral control +23/+67ms survives core-yield) PROVES marker is ON the T4 critical path;
the T1 self-test FLAT (+16/+8ms) validates the instrument AND confirms no T1 bootstrap. Good.

The DEFECTIVE step is "gzippy marker compute (59-139ms) EXCEEDS the ~51ms gap, THEREFORE rg
pays comparable." That inference only rules out rg=0. It is fully consistent with gzippy=100ms /
rg=30ms — i.e. ~70ms of gzippy-EXCESS marker — which would make marker a partial lever. "exceeds
the gap" is a consistency check, NOT a proof of shared/equal cost. Downgrade it.

The conclusion is nonetheless adequately supported by the SECOND argument, which IS sound: T1
carries no bootstrap, T4 adds it, and the output-removed ratio is invariant T1->T4 (0.867 ≈
0.862). If the bootstrap were net gzippy-excess it would WIDEN the ratio across that transition;
it does not. Honest caveat: this bounds ALL T4-incremental terms JOINTLY (bootstrap +
parallelism overhead), not the marker alone — but it does bound the marker net-excess to be
small. FIX: rest the verdict on ratio-invariance, not on "exceeds the gap." Conclusion (marker
is not the >=0.99 lever) holds.

---

## CLAIM 3 — PER-CHUNK/CHUNK-0/PIPELINE = the gzippy-specific excess (124ms/0.867x @T1).
## VERDICT: SOUND that it is REAL, gzippy-specific, and a LOWER bound; FIX-NEEDED on the LABEL.

Real and clean: output removed on both sides, T1 has no bootstrap, so by construction the 124ms
residual is gzippy's own ISA-L-invocation + pipeline. No output leak (gz removed it fully; the
iovec build is still paid, so it is not even discounted). The "/dev/null asymmetry => LOWER
bound" direction is CORRECT: gz over-removes (whole syscall) vs rg-null, so a symmetric removal
would only WIDEN the gap; 124ms is a robust floor (asymmetry magnitude itself sub-ms).

The defect is the NAME. 124ms / 16 chunks = ~7.75 ms/chunk. That is FAR too large for the named
per-chunk constants: `isal_inflate_init`/`set_dict` are µs, the prefetch/postprocess maps are
EMPTY at T1 (audit's own T1 weighting), chunk-0 is one-time, the format!/clock reads are µs.
Expressed per-BYTE the 124ms is gz 226 MB/s vs rg 261 MB/s on the SAME 211 MB — a ~13%
per-BYTE rate gap, the signature of driving the IDENTICAL ISA-L kernel sub-optimally, not a fixed
×16 per-chunk constant. Leading mechanisms: the ISA-L audit's non-LTO'd / non-inlined FFI glue
(per-return call overhead amortized across the chunk) + D1's 8× oversized output reserve causing
cache/TLB pressure that slows the kernel even at T1 (single reused buffer, but oversized). FIX:
re-label "per-chunk + per-BYTE ISA-L-invocation overhead," and the sizing oracle MUST split
per-chunk-constant from per-byte-rate (vary chunk size/count; compare gzippy-driving vs a direct
one-shot ISA-L call on the whole stream; toggle the D1 reserve).

LOAD-BEARING UNVERIFIED PREMISE (flag): the entire "engine matched at T1" attribution requires
rg 0.16.0 to use ISA-L at T1. Asserted from STATE.md / pipeline-audit, not re-measured this turn.
rapidgzip ships WITH_ISAL by default, so this is very likely true — but if rg's T1 ran its own
inflate, part of the 124ms would be ENGINE, not invocation. Verify first-hand before banking.

---

## CLAIM 4 — RECONCILE audit ("faithful, no big deviation") vs wall (13% per-chunk excess).
## VERDICT: CONSISTENT, not contradictory — resolved by separating STRUCTURE from COST.

The tension dissolves on one distinction: "per-chunk FFI handoff MATCHED to rg" means matched in
STRUCTURE (both call init + set_dict + boundary Vec per chunk), NOT matched in COST. The ISA-L
audit found gzippy's FFI glue is NOT LTO'd/inlined while rg compiles ISA-L into the same LTO unit
— so the identical structural operation is cost-heavier in gzippy. Faithful structure + a 13%
constant/per-byte cost excess coexist with no contradiction, because the excess is a
constant-FACTOR on a matched stage, not a missing or extra stage. The pipeline audit's own
HEADLINE already lands here ("more consistent with a diffuse per-chunk constant-factor than one
structural divergence"; "the per-chunk ISA-L inner-call constant factor... both pay, but gzippy
may pay a heavier instance"). That IS the structural oracle's per-chunk/pipeline term — they
agree.

ONE CORRECTION to the reconciliation as the ledger states it: do NOT claim "the diffuse D1-D7
SUM to 124ms." At T1, by the audit's own weighting, D2/D3 maps are empty, D4/D6 are µs, D5/chunk-0
is one-time, D1 reuses one buffer — they are µs-to-low-ms, they do NOT add to 124ms. The 124ms is
better explained by the ISA-L audit's non-LTO'd FFI glue (per-byte) + D1's oversized-buffer
per-byte cache cost than by the diffuse D-list. State it as a per-byte ISA-L-driving cost, not a
sum of the named per-chunk constants. With that wording, audit and wall are fully reconciled.

---

## CLAIM 5 — VERDICT "BAR-1 low-T unreachable = leading hypothesis, NOT floor-proved;
## per-chunk/pipeline is a candidate lever; next step = isolation oracle."  VERDICT: SOUND.

Correct and correctly hedged. Output is removal-proved shared; marker is bounded-shared by ratio
invariance; the residual collapses to a gzippy-specific 124ms that is NOT proved irreducible. So
"native pure-Rust >=0.99 every T is unreachable" is the LEADING HYPOTHESIS, not a proved floor —
right call.

The isolation oracle is the right AND necessary next move precisely because claim 3's label is
under-determined (per-chunk constant vs per-byte ISA-L-driving rate). The irreducibility
objection does NOT hold for gzippy-ISAL: it already links ISA-L, so the non-LTO'd glue is a
build-model fix (cross-language LTO, inline the hot setup, or feed ISA-L larger avail_out to
amortize per-return overhead) — rg is the existence proof that the same per-chunk structure can be
cheap. So this is a live, faithful lever for the ISAL build.

SCOPE caveat (the owner holds it; reaffirm): closing this helps the gzippy-ISAL build's T1/T4.
The PRODUCTION goal (native pure-Rust, C-FFI off the decode graph) ALSO faces the 0.667x VAR_VIII
engine wall, which IS bench-bounded. So for the NATIVE goal, BAR-1 at low-T is much closer to a
proved floor (engine-bounded) INDEPENDENT of this lever — the per-chunk lever does not rescue
native. Keep the two builds' BAR-1 verdicts separate.

---

## BOTTOM LINE
- Per-chunk/pipeline IS a real candidate lever worth a sizing oracle — for gzippy-ISAL. Not
  proved irreducible; build-model-closeable; rg is the existence proof.
- The audit-vs-wall tension is RESOLVED: faithful STRUCTURE + ~13% COST excess from non-LTO'd
  FFI glue (+ D1 oversized-buffer per-byte cost), not a missing stage. Do NOT bank it as
  "D1-D7 sum to 124ms."
- Two fixes before banking: (2) rest the marker verdict on ratio-invariance, drop "exceeds the
  gap" as proof; (3) re-label the residual "per-chunk + per-BYTE ISA-L-invocation overhead" and
  design the oracle to split per-chunk-constant vs per-byte-rate.
- One premise to verify first-hand: rg 0.16.0 uses ISA-L at T1 (else part of 124ms is engine).
- Claims 1 and 5 SOUND as written; the overall capstone verdict stands.
