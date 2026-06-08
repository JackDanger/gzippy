# DISPROOF VERDICT — ring→data drain isolation (GZIPPY_FOLD_NODRAIN)

Independent, read-only. Source-verified first-hand against this worktree
(`/home/user/www/gzippy-reimplement-isal`, branch reimplement-isa-l, HEAD
fc7336c3 + the 2 isolation knobs). I attacked all 5 angles and tried to break C1/C2/C3.
I did NOT re-run the bench; I assessed the VALIDITY of the reported signal and the
correctness of the instrument from the code.

## Summary up front (load-bearing finding first)

**The biggest disproof risk — that the no-op-drain removes legitimate work OTHER than
the drain memcpy, or distorts timing via cache/page state — does NOT hold, and where it
bites it makes +0.067× CONSERVATIVE, not inflated.** The no-op-drain (`writable_tail_reserve`
+`commit` over uninitialized space, gzip_chunk.rs:905-911) removes exactly the
`extend_from_slice` second-touch write of `chunk.data` (the term `ocl_cf` doesn't pay),
and nothing else: the per-clean-byte CRC still reads the *ring slice* `bytes`
(gzip_chunk.rs:899-903), the engine ring-write + back-ref resolution are untouched
(marker_inflate.rs:761-771), and downstream length accounting stays consistent. The one
side effect — `chunk.data`'s clean tail is left uninitialized — corrupts BYTES but not
WORK COUNT, and the only cache asymmetry it creates (cold `chunk.data` in the subsequent
output `writev` and the 32 KiB window-publish read) costs nodrain EXTRA, so it understates
the drain saving. **Verdict: C1/C3 UPHELD; C2 UPHELD-WITH-CAVEATS; refactor (C5) UPHELD.**
The single honest caveat is the charter one the brief already concedes: nodrain is a
measurement-only PROXY that BOUNDS the recoverable — it is not the banked number, and per
Measurement Rule 3 the ceiling is only set by actually REMOVING the region (the byte-exact
copy-free-to-final refactor), which is the owed next step.

---

## Angle 1 — best-of-N non-overlap vs high within-pass spread; freq-neutral control — UPHELD

The metric is best-of-11 per pass (the jitter floor). Best-of-N is the *correct*
estimator here because the noise is one-sided: CPU/memory-bus contention and scheduling
only ADD wall, never subtract it, so the per-pass minimum converges on the true floor and
the 28–76% within-pass spread is exactly the additive tail that best-of-N is designed to
reject. High spread therefore does NOT invalidate the comparison.

The separation is strong: native ∈ [0.1807, 0.1831] (width ~0.0024), nodrain ∈ [0.1645,
0.1684] (width ~0.0039), gap between ranges ≈ 0.0123 — roughly 3–5× the wider range's own
width, and 4/4 independent passes show native > nodrain with **no overlap**. That is a
sign-stable, reproducible signal, not a coin-flip artifact.

Freq-neutral soundness: the charter's preferred sleep-vs-spin control (Rule 2) does not
literally apply because nodrain is WORK-REMOVAL, not spin-INJECTION — so the load sweep is
the right substitute. Two controls defend it: (a) measure.sh is INTERLEAVED, so both
binaries see identical per-trial contention (the dominant control — it removes between-run
drift); (b) the nodrain/native RATIO is flat across load 1.21→2.33 (1.083–1.109×), whereas
a turbo artifact would SHRINK as load rises and turbo headroom disappears. The drain is a
memory-bandwidth op (low turbo sensitivity to begin with). **Caveat:** the load-invariance
is INDIRECT evidence, not a pinned-frequency measurement; a residual few-ms turbo component
cannot be fully excluded. But it is bounded and points the same direction as the interleave.
**UPHELD.**

## Angle 2 — does the no-op-drain skip OTHER legitimate work / change cache state? — UPHELD (nodrain is conservative)

Traced the post-drain reads of `chunk.data`:
- **CRC** (gzip_chunk.rs:899-903) reads the *ring slice* `bytes`, never `chunk.data`, so
  CRC cost is identical with/without the drain. nodrain removes ONLY the
  `extend_from_slice` write of `chunk.data` (the dest second-touch).
- **Window-publish** `last_32kib_window_vec` / `last_32kib_window` read from `self.data`
  (chunk_data.rs:485-488, 534-536) — i.e. `chunk.data`. Under nodrain this reads
  uninitialized bytes. But this read happens in BOTH paths (same 32 KiB copy cost), so it
  is timing-neutral; the only difference is the real path's drain pre-warmed those lines,
  so nodrain's read is slightly COLDER → costs nodrain more, not less.
- **marker-resolve / apply_window** gathers over `dataWithMarkers` using a window; a garbage
  window changes the RESULT bytes but not the gather COUNT (driven by marker positions), so
  timing-neutral.
- **Output `writev`** streams `chunk.data` to the writer within the measured wall. nodrain
  reads cold/possibly-page-faulting-on-read pages (Vec::reserve gives untouched pages,
  segmented_buffer.rs:206-217); the real path faulted/warmed them during the drain write.
  So the page-fault + cold-read cost is SHIFTED into nodrain's writev, partially offsetting
  the saving.

Net: every asymmetry the no-op-drain introduces works AGAINST it (extra cold reads), so it
UNDER-states the true drain cost. It does not accidentally elide any compute work. The
drain memcpy is cleanly isolated. **UPHELD — and the bias is conservative.**

## Angle 3 — full decode ran before the trailer check; 2 seeded chunks don't distort — UPHELD

- The terminal failure is the *combined-CRC trailer* verify (crc32.rs:226-254 `verify` /
  `combine_crc32`), which runs only AFTER all chunks decode, apply_window resolves, windows
  publish, and bytes stream. So the measured wall IS the full decode+output wall; the exit-1
  is a post-decode check, not an early abort (matches the self-test diagnosis).
- Why it fails under nodrain: the 2 window_seeded chunks read their initial window from
  `chunk.data` (garbage), and the finished_no_flip chunks resolve markers against the
  predecessor (garbage) window — both produce wrong bytes → wrong combined CRC. This is the
  expected byte-corruption signature, and it confirms the knob fires WITHOUT changing decode
  work count.
- `worker.isal_stream_inflate` (gzip_chunk.rs:586-600) is a misnamed span wrapping
  `StreamingInflateWrapper` (pure-Rust). Real ISA-L FFI only runs under
  `GZIPPY_ISAL_ENGINE_ORACLE=1`, which gates `ISAL_ENGINE_ORACLE_CHUNKS`
  (gzip_chunk.rs:308); with that counter 0, no FFI ran. The 2 seeded chunks are pure-Rust
  clean continuations and do the same decode work in both binaries, so they cannot distort
  the delta. **UPHELD.**

## Angle 4 — is ~0.11× a sound UPPER BOUND on the remaining intrinsic+ring-write gap? — UPHELD-WITH-CAVEATS

The arithmetic (0.188× residual − ~0.067× drain = ~0.121×) is the right *method* (oracle-
removed ceiling minus a removed term, Rule 3-shaped). The remaining ~0.11× still CONTAINS
the engine ring-write (nodrain does not touch marker_inflate.rs:761-771), so the
intrinsic-symbol-rate component is strictly ≤ 0.11× — a genuine upper bound on symbol rate.

Caveat 1: the ≤0.11× sharpening is valid only if the drain saving is AT LEAST ~0.067×. If
+0.067× were turbo-inflated (drain saving smaller), the true remaining would be LARGER and
the ≤0.11× bound would fail. Two things rescue it: (a) angle-2's cold-cache bias makes the
measurement conservative (true drain ≥ measured); (b) the unconditional fallback statement
"intrinsic symbol rate ≤ 0.188×" holds regardless. So the DIRECTION (the symbol-rate gap is
smaller than the prior 0.188× bound) is sound; treat 0.11× as a best-estimate upper bound,
not a proven one.

Caveat 2: `ocl_cf` is still a DIFFERENT engine (ISA-L) AND ring-free AND copy-free-to-final
(per the prior fold-contig verdict L2). nodrain removes only the drain, not the ring-write
and not the engine difference. So 0.11× still confounds {intrinsic pure-Rust symbol rate} +
{ring-write}. The only way to fully strip it is a same-engine pure-Rust ring-FREE oracle —
which is the refactor itself. **UPHELD-WITH-CAVEATS.**

## Angle 5 — is the copy-free-to-final refactor the faithful next step? does vendor decode the clean tail into one contiguous buffer with no ring? — UPHELD

Vendor evidence (DecodedData.hpp): the u16 marker ring (`dataWithMarkers`) exists ONLY for
the pre-window bootstrap; the comment at 278-281 states explicitly that "as soon as we have
32 KiB of symbols, the decompression should delegate to ISA-L" — i.e. the clean BULK is
decoded straight to u8 `data`, never through the u16 marker ring + narrow pass. So gzippy's
ring→`chunk.data` drain memcpy for the CLEAN phase has NO vendor counterpart; eliminating it
(engine writes clean u8 directly into `chunk.data`'s reserved tail, back-refs resolve from
that contiguous tail) CONVERGES toward vendor. This also matches the governing memory
direction (u8-direct clean = the same engine, the u16 ring is the shortcut/deviation).

One honest nuance so the refactor isn't over-sold: vendor's `appendDecodedData`
(DecodedData.hpp:282-289) DOES copy its `data` buffer views into one contiguous `copied`
buffer at merge time — so "literally zero copies" is not vendor-true. The faithful claim is
narrower and correct: vendor has **no u16-ring + per-block narrow/drain for the clean
phase**, and removing gzippy's is convergence, not innovation. **UPHELD.** The refactor is
the right next step AND the only thing that turns the +0.067× from a measured BOUND into a
banked, byte-exact recovery (Rule 3 — a no-op proxy bounds, only removal banks).

---

## Per-claim verdicts

- **C1** (drain removal moves native_fold ~0.745×→~0.812× = ~+0.067× recoverable, sign-stable
  best-of-N over 4 non-overlapping passes): **UPHELD-WITH-CAVEATS.** Signal is valid (best-of-N
  on one-sided noise, interleaved, load-invariant); the cold-cache asymmetry makes it
  conservative. Caveat: it is a measurement-only BOUND, not the bankable number, and a small
  turbo component can't be fully excluded without a pinned-frequency run.
- **C2** (residual splits into ~0.067× drain + ~0.11× UPPER BOUND remaining = intrinsic + ring-write):
  **UPHELD-WITH-CAVEATS.** Right method and direction; ~0.11× still confounds ring-write +
  engine-difference, so it is an upper bound on symbol rate, not symbol rate. The
  unconditional safe statement is intrinsic ≤ 0.188×; ≤0.11× is the best-estimate sharpening.
- **C3** (CRC is not a lever, nodrain_nocrc ≈ nodrain, Δ<0.3%): **UPHELD.** Corroborated by
  code — CRC reads the ring slice (a small compute add), independent of the drain write.
- **C5 / refactor faithfulness** (copy-free-to-final, no ring for clean phase, vendor existence
  proof): **UPHELD.** Vendor decodes the clean bulk to contiguous u8 with no u16 ring; the
  drain has no vendor counterpart. Nuance: vendor still concatenates at merge time, so claim
  "no clean-phase ring/narrow," not "zero copies."

## Recommendations
1. Proceed to the byte-exact copy-free-to-final refactor — it is faithful (angle 5) and is
   the only way to BANK C1 (the no-op proxy only bounds it). Re-measure on `scripts/measure.sh`
   interleaved + sha-verified vs rg; the banked number replaces +0.067×.
2. When stating C2, lead with the unconditional "intrinsic symbol rate ≤ 0.188×" and present
   ≤0.11× as the drain-removed best estimate (still inclusive of the ring-write).
3. If a referee disputes the turbo caveat (angle 1/4), a single pinned-frequency
   (turbo-off, fixed-freq) pass would convert the load-invariance argument from indirect to
   direct — cheap insurance, not required.

=== ADVISOR EXIT 0 ===
