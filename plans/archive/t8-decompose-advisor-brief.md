# DISPROOF-ADVISOR BRIEF — T8 seeding-bundle decomposition

You are an INDEPENDENT, READ-ONLY disproof advisor. Your job is to try to BREAK the
owner's decomposition before it is relied on. Source-verify against the repo at
/Users/jackdanger/www/gzippy-reimplement-isal (branch reimplement-isa-l). Do NOT build,
do NOT measure, do NOT edit. Write your verdict to
plans/t8-decompose-advisor-verdict.md and nothing else.

## CONTEXT
Phase-0 (plans/asmport-phase0-advisor-verdict.md, upheld): pure-Rust ENGINE already TIES
rapidgzip at T8 once GZIPPY_SEED_WINDOWS is on (~1.0×); unseeded prod LOSES (~0.66×). The
advisor flagged that GZIPPY_SEED_WINDOWS bundles THREE removals: (a) u16 marker-COMPUTE,
(b) block-finder real-boundary vs partition-guess ALIGNMENT, (c) the 13 spec-failure
re-decodes. The owner was tasked to DECOMPOSE the bundle before choosing the Phase-1 fix.

## WHAT THE OWNER DID (verify the code)
Added 2 measurement-only env knobs (OFF==identity), src/decompress/parallel/:
- `seed_windows.rs::seed_no_windows()` (GZIPPY_SEED_NO_WINDOWS=1): makes `seed_window_for`
  always return None (suppress the seeded-window fallback). Used at seed_window_for().
- `seed_windows.rs::seed_no_boundaries()` (GZIPPY_SEED_NO_BOUNDARIES=1): skips the
  block_finder pre-seed loop at chunk_fetcher.rs:~498. Keeps windows.
Cells (all WITH GZIPPY_SEED_WINDOWS): seedfull (both), onlywin (NO_BOUNDARIES = windows
only), onlybnd (NO_WINDOWS = boundaries only), prod (no seeding). All byte-exact sha
028bd002…cb410f.

## RESULTS (full data in plans/t8-decompose-findings.md; pre-reg in plans/t8-decompose-prereg.md)
Walls (2 runs): seedfull ~0.13s TIE; onlywin/onlybnd/prod all ~0.198s ~0.66× LOSS,
indistinguishable. Counters: seedfull window_seeded=17 spec-fail=0 Fill 91% decodeBlock
0.846s; onlywin seed_hits=0 (windows UNUSED!) window_seeded=2 spec-fail=13 decodeBlock
1.06s ≡ prod; onlybnd spec-fail 13→0 but body still 170MB/s u16 decodeBlock 1.106s.
rapidgzip --verbose window-absent same 34.5% markers: decodeBlock 0.542s (≈2× faster than
gzippy's window-absent 1.067s).

## OWNER'S VERDICT (attack it)
Pinpointed sub-lever = **marker-COMPUTE**: gzippy's window-absent u16 decode is ~2× slower
than rapidgzip's u16 marker decode on the SAME workload (1.07s vs 0.54s decodeBlock).
Boundary-alignment (b) removes spec-failures but is WALL-NEUTRAL (onlybnd ≈ prod wall);
spec-failures (c) are not dominant. Ceiling = the T8 1.0× TIE (seedfull oracle).

## QUESTIONS FOR YOU (genuine disproof)
1. Is the "windows unusable without seeded boundaries (seed_hits=0 in onlywin)" claim
   correct from the code? (seed store keyed by real boundaries; prod dispatches at
   partition guesses ⇒ seed_window_for misses.) Does this make onlywin a VALID isolation
   of boundaries, or does it confound the decomposition (windows-only is unrealizable)?
2. Is "onlybnd removes spec-failures yet is wall-neutral ⇒ boundary-alignment is not the
   dominant lever" sound, or could onlybnd's marginally-WORSE decodeBlock mask a real
   boundary benefit?
3. Is the rapidgzip apples-to-apples (rg marker decodeBlock 0.54s vs gzippy window-absent
   1.07s, both window-absent, same 34.5% markers) a FAIR comparison, or are the --verbose
   "decodeBlock SUM" semantics different between the tools? (Check both are summed over
   workers / same denominator.)
4. Is the ceiling claim ("fixing marker-COMPUTE ⇒ T8 TIE") a valid PROCESS#3 removal
   oracle, or does seedfull remove MORE than the marker-compute (e.g. also applyWindow
   serial work that rg pays)? Does that make the ceiling OPTIMISTIC?
5. Any other disproof angle. State which sub-lever the data actually supports and whether
   the owner's "marker-COMPUTE dominates" is UPHELD / UPHELD-WITH-CAVEATS / REFUTED.
