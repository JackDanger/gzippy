# DISPROOF-ADVISOR VERDICT — Phase-0 ISA-L-in-pipeline WALL oracle

Independent, read-only. Source-verified against:
`src/decompress/parallel/gzip_chunk.rs` (oracle + gate),
`src/decompress/parallel/seed_windows.rs`,
`src/backends/isal_decompress.rs` (`decompress_deflate_from_bit_with_boundaries`),
`src/decompress/parallel/chunk_fetcher.rs` (`run_decode_task` + block_finder seed).

---

## CLAIM 1 — oracle measures an igzip-class engine in the REAL pipeline, byte-exact
**Verdict: UPHELD-WITH-CAVEATS.**

Source-confirmed:
- The oracle is gated INSIDE `finish_decode_chunk_impl` (gzip_chunk.rs:534), the
  clean-tail decode entry. Pool/consumer/ring/window-publish/scheduling are not
  touched — only per-chunk byte production is swapped.
- It feeds ISA-L's output through the SAME `ChunkData` primitives the pure path
  uses: `writable_tail`/`commit`/`note_inner_decoded_bytes` (gzip_chunk.rs:243-251),
  per-byte CRC over the kept region (252-257), `append_block_boundary_at` replay
  (263-271), `finalize_with_deflate` (273). This matches the pure path's accounting
  (compare 698-746).
- The FFI is genuine patched igzip with real `ISAL_STOPPING_POINT_END_OF_BLOCK`
  boundary recording (isal_decompress.rs:666, 755-768); boundaries are computed
  from `avail_in`/`read_in_length` at a real block transition — provably real.

Caveats (do not change the verdict, but bound it):
1. **The oracle only replaces the CLEAN tail (full 32 KiB window present)** —
   gated by `initial_window.len() != MAX_WINDOW_SIZE ⇒ Ok(false)`
   (gzip_chunk.rs:175-177). It NEVER replaces the marker-phase (window-absent) u16
   decode — ISA-L cannot emit u16 markers. So "igzip-class engine in the pipeline"
   is true for the *bulk clean decode only*, which is exactly the region seeding
   makes universal. Fine for the stated purpose; see Claim 4.
2. **The run is "contaminated" by the brief's OWN standard.** `isal_oracle_fallbacks=1`
   ⇒ 1/17 clean chunks decoded by pure-Rust, not ISA-L. The doc comment at
   gzip_chunk.rs:131-136 says any non-zero fallback means the wall "must be
   discarded." Strictly, the isal number is impure. It does not flip the verdict
   (1 chunk ≈ 6%, and the fallback chunk is decoded by the pure engine which itself
   ties), but the brief should not call this "94% real ISA-L" and also rely on its
   own "discard if non-zero" rule — pick one. Treat the isal number as
   "≈ TIE, ±1-chunk contamination," not a clean measurement.

---

## CLAIM 2 — an igzip-class engine ALONE does not close the T8 gap (TIE-vs-TIE)
**Verdict: UPHELD-WITH-CAVEATS (sound at T8 only).**

The load-bearing fact is not the isal oracle — it is that **`pure` (the SLOWER
engine) already ties rg once seeded** (1.002). If the slower engine ties when the
marker path is removed, the engine cannot be the T8 wall binder. The isal oracle is
confirmatory: a faster engine cannot beat a tie, and in fact measured slightly
*worse* (0.905) — consistent with FFI/re-CRC/64 MiB-alloc/256 KiB-overshoot
overhead, NOT with ISA-L being slow. Either way the engine swap does not move the
wall past the tie pure already reaches.

Disproof angles checked and cleared:
- **Bounded slice making ISA-L artificially fast?** No — the opposite. ISA-L decodes
  to the first boundary at-or-past `stop_hint` within `[..stop_byte+256KiB]`
  (gzip_chunk.rs:189-237), which can *overshoot* the natural stop into the slack and
  then trim (`keep_len`). That is EXTRA work, penalizing isal, not flattering it.
  Under-decode is impossible: a short slice ⇒ END_INPUT mid-block ⇒ `None` ⇒
  counted fallback (isal_decompress.rs:736-744). sha-identical output confirms no
  bytes skipped.

Caveats:
1. **This is a T8-scoped result.** The engine gap is REAL (decodeBlock SUM
   pure-seed 0.781 vs rg 0.517 ≈ 1.51×) and is wall-neutral ONLY because Fill is 90%
   — parallel slack absorbs the per-thread rate deficit. At T1 (no slack, engine
   rate binds the wall directly) or T16 (more contention) it may bind. The asm
   port's value is NOT refuted there; it is simply not the T8 lever.
2. **gzippy-seeded does LESS total work than rg yet only TIES.** rg pays for windows
   via its serial `applyWindow` resolution; gzippy-seeded gets correct windows for
   free and skips resolution entirely. A work-equal engine doing less work should
   *beat* rg — it only ties. That is independent evidence the pure engine is ~1.5×
   slower per byte (consistent with the isolation number) and is being *masked* by
   the free-window shortcut, not absent. "Engine is not the binder" is true at the
   wall; "engine is at parity" is FALSE.

---

## CLAIM 3 — the T8 binder is the window-absent marker/speculation path
**Verdict: UPHELD as a coarse localization; the conclusion is SOUND, the seeding is
NOT unfair, but it is CONFOUNDED (bundles three removals into one knob).**

This is the central question. The seeding is a valid causal perturbation per
Measurement PROCESS #3 (REMOVE the region, measure the wall): removing the
window-absent path moves the wall 0.69→1.00 — a large, monotonic response ⇒ the
region is on the critical path. That inference is sound.

**Is seeding an unfair oracle / does rg get windows cheaply some other way?**
No, it is not unfair *for localizing the binder*, and the asymmetry strengthens the
conclusion rather than breaking it. rg does NOT get a free correct window — rg
speculatively decodes with u16 markers (the genuine 34.5% replaced symbols) and
resolves them in a serial `applyWindow` pass after the real predecessor finishes.
gzippy-seeded gets the correct predecessor window handed to it (captured from a
prior real decode, seed_windows.rs:81-90, replayed at chunk_fetcher.rs:2269-2273)
and so does *less* work than rg. gzippy-prod, doing its own marker work, loses
0.69×. So gzippy's marker path is slower than rg's marker path — apples-to-apples on
the SAME 34.5% marker workload that rg ties on. Sound.

**The load-bearing caveat — seeding removes a BUNDLE, not "the marker engine":**
Seeding simultaneously does THREE things, only one of which is the marker decode:
1. Provides the correct predecessor window ⇒ skips u16 marker decode + resolution.
2. **Pre-seeds the block_finder with the REAL boundaries** (chunk_fetcher.rs:499-504
   via `seedable_chunk_starts`) ⇒ every dispatch lands on a true boundary instead of
   a partition GUESS. Prod gets NO such pre-seed (`seedable_chunk_starts` returns
   empty off-seed, seed_windows.rs:239-241).
3. Eliminates the 13 header-speculation failures and the re-decodes/head-of-line
   stalls they cause ⇒ Fill 77→90.

The 0.69→1.00 wall delta is attributable to this whole bundle. It is NOT isolated to
"the u16 decode rate." In particular, the project's own recorded finding
(`project_confirmed_offset_prefetch_gap`: ~40% of the T8 wall is 4 head-of-line
stalls from partition-guess misalignment, "fixable, NOT architectural") suggests a
large share of the delta is item (2) — boundary MISALIGNMENT / scheduling — which
seeding fixes for free via block_finder pre-seeding, and which has nothing to do
with how fast any engine decodes markers. So:

- "The binder is the window-absent path" — UPHELD (the bundle is on the critical path).
- "The binder is the marker-decode COMPUTE specifically" — NOT ESTABLISHED. The
  oracle cannot separate marker-compute from boundary-alignment from
  speculation-failure re-decodes. Phase-1 must perturb these independently (e.g.
  seed ONLY the block_finder boundaries WITHOUT the windows, vs seed ONLY windows
  with prod boundaries) to decompose.

---

## CLAIM 4 — asm engine port can't move the prod T8 wall; target the window-absent path
**Verdict: UPHELD-WITH-STRONG-CAVEATS, leaning REFUTED as stated.**

The directional recommendation — *at T8, prioritize the window-absent path over the
asm engine port* — is reasonable and follows from Claims 2+3. But the stronger
inference "the asm engine port cannot move the prod T8 wall" is NOT established and
is likely wrong for two reasons:

1. **The asm/faster-kernel techniques also accelerate the marker-phase decode, which
   IS on the binding path.** The oracle replaced ONLY the clean tail with ISA-L; the
   window-absent u16 marker bootstrap still ran gzippy's pure-Rust kernel at
   168 MB/s (per the brief's own --verbose). If any meaningful fraction of the
   0.69× is marker-decode RATE (not alignment/scheduling), a faster inner kernel
   adapted to u16 output would help the binding path. The oracle never tested this —
   ISA-L can't emit markers — so the marker-decode rate's contribution to the wall
   is UNMEASURED. Claim 4 treats "engine port" as synonymous with "clean path only,"
   which the code does not support: the same primitives feed both phases.

2. **The T1 case is unaddressed.** At T1 there is no Fill slack; the 1.51× engine gap
   would bind the wall directly. The asm port is plausibly the T1 lever. The brief's
   "at least at T8" hedge is honest, but the campaign goal is parity across
   T1–T16, so "deprioritize the asm port" must not be read as "abandon it."

**Net for phase-1:** the highest-value next perturbation is to DECOMPOSE the Claim-3
bundle — separate boundary-alignment (block_finder seed only) from marker-compute
(window seed only) — before committing to either the asm engine port OR a marker-path
rewrite. If the wall delta is mostly alignment (consistent with the recorded
head-of-line-stall finding), neither the asm engine nor a faster marker kernel is the
T8 lever — the block finder / prefetcher is.

---

## SUMMARY
| Claim | Verdict |
|-------|---------|
| 1 — oracle measures igzip-class engine in real pipeline, byte-exact | UPHELD-WITH-CAVEATS (clean-tail only; 1-chunk fallback impurity) |
| 2 — engine alone doesn't close T8 gap (TIE-vs-TIE) | UPHELD-WITH-CAVEATS (T8 only; engine gap real but slack-masked, NOT at parity) |
| 3 — binder is the window-absent marker/speculation path | UPHELD as coarse localization; sound & not unfair, but CONFOUNDS marker-compute with boundary-alignment + speculation re-decode |
| 4 — asm port can't move prod wall; target window-absent path | Directional recommendation UPHELD at T8; strong inference REFUTED — marker-phase decode rate is on the binding path and was never replaced; T1 unaddressed |

**Most-important question:** "An igzip-class engine alone does NOT close the T8 wall;
the binder is the window-absent marker path" — the FIRST half is a sound conclusion
(the slower pure engine already ties when seeded; valid causal perturbation). The
SECOND half is sound only as a *coarse* localization to the window-absent BUNDLE; it
is NOT established that the marker DECODE (vs boundary misalignment / speculation
re-decodes) is the dominant sub-binder. The seeding oracle is not unfair — it removes
a region and the wall responds — but it removes three things at once and gives gzippy
a free correct window that rg earns serially, so it cannot attribute the gain to any
single sub-component. Decompose the bundle before choosing the phase-1 lever.
