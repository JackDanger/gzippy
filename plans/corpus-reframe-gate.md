# DIS-28 corpus-generality reframe — disproof gate (read-only advisor)

Binary b9eb0a73, frozen guest, interleaved N=5, sha-verified, path-asserted.
Source verified against HEAD d56cb0f5 (same decode files; mechanism unchanged).
ratio = rg/gz, >1 gz wins, bar TIE>=0.99.

## Per-claim verdict

### (a) The REFRAME — SOUND, with precision fixes
- **"engine-W is the corpus-general root": SOUND.** The cleanest evidence is the
  `model` corpus (1.26x near-incompressible, 51 chunks, flip=49): it loses at
  **every** T (T1 0.89 / T8 0.685 / T16 0.677). Because ratio 1.26 << 8 it does
  NOT hit the reserve storm (below), and flip=49/51 means nearly every chunk runs
  the pure-Rust marker+clean engine. So model's 0.68 is genuinely the
  chunk-count-amplified engine speed gap, not a confound — REFUTED-attempt failed,
  claim survives.
- **"high-T loss is silesia-specific": IMPRECISE / partly REFUTED.** model loses
  HARDER at T16 (0.677) than silesia (0.92), so "high-T loss" is not silesia-only.
  The defensible version — and what the data supports — is: **engine-W is a
  general deficit visible across all T, not a T16-scheduler artifact.** State it
  that way, not as "silesia-specific."
- **"deficit scales with compressed-size / chunk-count": PLAUSIBLE, not
  established.** It is a 2-point trend (model largest+worst, silesia mid). Sign is
  consistent; N=5 corpora do not pin the functional form. Keep as hypothesis.
- **small 6MB win (1.74-1.89x): REAL number, but a STARTUP artifact, NOT engine.**
  It wins at ALL T including T16 despite isal_chunks=0 (all-fallback) — i.e. the
  engine is in its WORST mode there yet still wins, which can only be the ~34ms vs
  ~64ms startup delta on a tiny file. Do not bank it as a compressible-engine win.

### (b) The FALLBACK STORM — REAL (source-confirmed), but MECHANISM MISATTRIBUTED
- **Storm is REAL and source-explicable. CONFIRMED.** `decompress_deflate_from_bit_into`
  (src/backends/isal_decompress.rs:863-869) returns `None` — deliberately does NOT
  realloc — the moment the caller's reserved `out` slice fills. The reserve is
  `compressed_span * EXPAND_FACTOR`, `EXPAND_FACTOR = 8`
  (src/decompress/parallel/gzip_chunk.rs:265-271). `None` → `Ok(false)`
  (gzip_chunk.rs:281-282) → `ISAL_ENGINE_ORACLE_FALLBACKS += 1` (gzip_chunk.rs:680)
  → byte-exact but ~7.5x-slower pure-Rust re-decode.
- **The dominant cause is UNDER-RESERVE at factor 8, NOT the EOB-stop / inexact
  contract the owner cites.** Smoking gun: the storm threshold is exactly 8.
  ghcn 7.8x → NO storm (T1 0.95); nasa 9.9x → storm (T1 0.57); small 10x → storm.
  The discontinuity sits precisely at EXPAND_FACTOR=8. The EOB-stop decline path
  (gzip_chunk.rs:312-330) fires only on STORED/FIXED-Huffman blocks that record
  zero boundaries — but nasa/small are highly-compressible DYNAMIC-Huffman, so
  that path does not apply to them. The owner's own comment (gzip_chunk.rs:663)
  already lists "a pathologically compressible chunk overrunning the
  chunk-proportional reserve" as a distinct decline cause — that is this one.
- **"Same fix as JOB-2 (isal-resync-stored-fixed / SYNC_FLUSH): REFUTED.** JOB-2
  addresses the stored/fixed EOB-resync decline (the line 312-330 path). The storm
  is the reserve-overflow `None` path (line 281-282 / isal_decompress.rs:866-868).
  Different cause, different fix.
- **FIXABLE, and CHEAPER than the owner thinks.** No readStream-coalesce port
  needed. Either retry-on-`None` with a doubled reserve, or auto-size
  EXPAND_FACTOR from the running observed expansion ratio (cap stays for the
  footprint-oracle path only; production wall just pays a larger one-time alloc).
  This is a faithful correctness-preserving change (bytes already identical; only
  which engine emits them changes), and it directly targets nasa T1 0.566.

### (c) b9eb0a73 has no single-shot route — TAKE AS GIVEN, important caveat
- Accept claim (c): these T1 = ParallelSM-T1, not the banked single-shot win.
- **Consequence:** production T1 for nasa would route to single-shot ISA-L, which
  has no chunked reserve at all — so the storm's T1 damage (0.566) is partly a
  non-production benchmarking artifact. It is real in ParallelSM-T1, mooted if
  production T1 = single-shot.

### (3) Does the storm matter for the WALL? — MIXED; FIX-NEEDED to bound
- nasa T8/T16 WIN (1.04-1.05) DESPITE all-fallback because nasa is only ~3 chunks
  (20MB): 3 slow pure-Rust decodes spread across 8-16 cores hide the 7.5x tax.
- So on nasa it is effectively a T1-only lever (and possibly single-shot-mooted).
- **BUT it is NOT intrinsically T1-only.** For a LARGE >8x corpus where
  chunk_count >> thread_count, every thread serializes multiple 7.5x re-decodes,
  so the storm would tax parallel-T too. nasa is just too small to expose that.
  To bound the lever's wall value, re-measure on a >=200MB, >8x corpus at T8/T16
  before ranking it above the user-gated work.

### (4) CONVERGENCE — YES, owner-turnable work reopens
The campaign is NOT "only user decisions remain." Ranked next steps:
1. **Storm fix (retry-on-None / auto-size reserve).** Cheap, high-confidence,
   correctness-preserving, owner-turnable, removes a confound from the scorecard.
   Do FIRST. Caveat: confirm its parallel-T payoff on a large >8x corpus; on the
   current corpora its proven win is T1/small only.
2. Merge the banked single-shot + footprint wins (also owner-turnable; fixes the
   T1 representativeness gap in (c)).
3. engine-W / large-incompressible (model, silesia-T16) — the headline lever, but
   gated on the user asm decision.
4. The user TIE-bar decision.

## Accurate real-world scorecard
"gz-isal wins small/compressible, ties text, loses large-incompressible" is
DIRECTIONALLY right but the "compressible" cell is contaminated and must be split:
- **small inputs: WIN** — but startup-dominated, not an engine win.
- **large compressible (>8x): high-T WIN, low-T STORM-TAXED** — the low-T loss is
  the fixable reserve bug, not engine-W; provisional pending the large->8x re-measure.
- **text (silesia/ghcn): TIE at T8, slight LOSS at T16.**
- **large incompressible (model): LOSS at all T (0.68-0.89)** — the genuine
  engine-W, the real headline deficit.

## Bottom line
- The reframe is **BANKABLE** after two corrections: say "engine-W is a general,
  all-T deficit" (not "high-T loss is silesia-specific"), and treat the small win
  as a startup artifact.
- The fallback storm is a **REAL, source-confirmed, FIXABLE faithful lever**, but
  its mechanism is **reserve under-sizing at EXPAND_FACTOR=8**, NOT the EOB-stop
  contract and NOT the JOB-2 fix. The fix is correspondingly cheaper.
- It is the **highest-value cheap next step** ahead of the asm/bar decision (it is
  owner-turnable and removes a scorecard confound), but its wall payoff beyond
  T1/small is **unproven** — bound it on a large >=8x corpus at parallel T before
  claiming it moves the parallel wall.
