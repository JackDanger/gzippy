# MORNING BRIEF — overnight 2026-06-08/09 (supervisor consolidation)

## ============ FIRST PRODUCTION WIN BANKED (T1 single-shot, Opus-gated SAFE) [2026-06-09] ============
gzippy-isal T1: routed single-threaded single-member to single-shot ISA-L (DecodePath::IsalSingleShot,
cfg(parallel_sm,isal_clean_tail), num_threads<=1; BGZF/multi-member classified above it; native
unchanged). MEASURED frozen guest: T1 0.905x -> 1.200x rg (WIN, beats rg), byte-exact (887 lib tests,
dual-sha both features T1/T4/T8, multi-member-at-T1 -> MultiMemberSeq safe), T4/T8 unchanged. Opus gate:
all 5 claims SOUND, SAFE TO BANK. UPDATED isal scorecard vs >=0.99-every-T: T1 1.200x PASS / T8 1.038x
PASS / T4 0.906x = the ONLY failing cell (parallel-scheduling, pre-existing, asm-bounded Huffman-rate +
small scheduling slice). Code on worktree branch owner/t1-singleshot-route (uncommitted, gated-ready).
READY-TO-COMMIT set: T1 single-shot + JOB-2 reserve fix + build.rs comment fix (all gated-PASS).

## ============ DIAGNOSTIC PHASE COMPLETE (Opus-gated, DIS-21 + campaign-conclusion gate) ============
After exhaustive disproof, gzippy is STRUCTURALLY FAITHFUL to rapidgzip — every named structural/
scheduling/memory lever was refuted WITH A MECHANISM, and the engine kernel is the byte-identical
AVX2 nasm. The remaining gap is localized + the remaining items are DECISIONS, not open levers.

FINAL localized attribution (gated):
- isal T1 (0.90x): RECOVERABLE chunking-pipeline SERIALIZATION (markers~=0 at T1; single-shot ISA-L
  beats rg 1.197x — DIS-15). NOT the Huffman symbol rate. Lever = T1 routing (see decisions).
- isal T4 (0.91x): asm-bounded marker-prefix Huffman SYMBOL RATE (gzippy pure-Rust read_internal_
  compressed vs rg's ISA-L-AVX2 HuffmanCodingISAL primitive). The u16 output/backref/segment
  machinery is byte-for-byte faithful to rg (deflate.hpp:805 m_window16 IS a ring; resolveBackreference
  == emit_backref_ring; appendToEquallySizedChunks == SegmentedU16 drain) — de-frag was a PHANTOM
  (DIS-21, vendor-verified; flat-backref A/B = clean TIE, rule-3 low-ceiling). [NB: rg's output/backref
  share is a structural INFERENCE, not perf-isolated — but triangulated by source + the flat A/B + VAR_VIII.]
  + a bounded-small parallel-scheduling/handoff slice (DIS-16/17, attributed-after-refutation, not zeroed).
- isal T8 (0.990x): TIES (parallelism hides the symbol-rate gap).
- native: the above PLUS its own clean-path 0.667x asm engine floor (VAR_VIII).
The marker-prefix + clean-tail Huffman SYMBOL RATE (pure-Rust vs ISA-L AVX2) is the SAME inner-loop
asm ceiling VAR_VIII proved plateaus at 0.667x — and it is USER-GATED (below its own 0.85 bar).

REMAINING (DECISIONS + 1 correctness gap — NOT open diagnostic levers; do NOT churn more oracles):
- D-T1: route T1/single-threaded gzippy-isal to single-shot ISA-L (RECOMMENDED — clear byte-exact win,
  1.197x beats rg; faithful to the isal "hand off to ISA-L at the right spot" charter; BUT reverses the
  deliberate 5e563dc parallel-SM-everywhere choice + the ONE-PRODUCTION-PATH value => user ratifies the
  architecture reversal).
- D-asm: the T4/native Huffman symbol-rate lever = the full-kernel asm, which FAILED its own 0.85
  isolation bar (VAR_VIII 0.667x) => HOLD per the pre-registered gate, OR revisit funding it.
- D-bar: the >=0.99x-every-T bar itself — given the inner-Huffman asm plateau, low-T BAR-1 is likely
  unreachable for native pure-Rust; isal needs the T1 routing + the (gated) asm for T4.
- CORRECTNESS (ready to merge, gated-PASS): JOB-2 SYNC_FLUSH reserve fix (branch isal-resync-stored-fixed)
  + the build.rs comment fix (HEAD). OPEN-3 SYNC_FLUSH coverage is the orthogonal correctness gap.

## BOTTOM LINE
The gzippy-ISAL low-T gap is NOT a floor — it's the per-chunk ParallelSM pipeline overhead,
PROVEN recoverable: at T1, one ISA-L call (no chunking) = 1.197x rg (BEATS rapidgzip 20% with
the same igzip kernel). gzippy-NATIVE is separately walled by the 0.667x pure-Rust+asm engine
ceiling (likely can't hit >=0.99 at low-T).

## VERIFIED SCORECARD (gated, frozen guest, interleaved, sha-verified, vs rg 0.16.0)
isal T1 0.905 / T4 0.911 / T8 0.990 ; native T1 0.608 / T4 0.761 / T8 0.915.

## isal LOW-T — fully attributed (six gated refutations + the proof)
- T1 gap = 100% per-chunk ParallelSM pipeline (247ms/24%): mostly the SERIALIZATION tax (each
  chunk waits the prior's 32KB window before ISA-L runs => chunking buys NOTHING at 1 thread)
  + ring/window-map/CRC/handoff. NOT engine (==rg nasm), NOT placement (dead), NOT output
  (shared floor), NOT marker bootstrap (shared), NOT buffer over-reserve (page-faults
  written-page-only), NOT LTO/glue (kernel is nasm). FORCE_PARALLEL_SM is a DEAD no-op => the
  0.905x is real production -p1 (ParallelSM, no thread floor, mod.rs:170-188).
- PROOF it's recoverable: single-shot ISA-L (decompress_gzip_stream) @T1 = 1.197x rg, byte-exact.
- T4 0.911x = a SEPARATE parallel-scheduling gap. SIZED (DIS-16): the faithful consumer-lean
  (D2/D3/D4) is NULL/TIE (removed overheads are µs-scale, 2-3 orders below the ~55ms gap; the
  time is in fetcher_get ISA-L-decode-wait + postproc_dispatch, untouched by the lean). The T4
  gap is DEEPER parallel-SCHEDULING, and a new signal pins it: gzippy T4 run-to-run variance is
  17-36% vs rg's 8-10% => scheduling/CONTENTION nondeterminism. This is the campaign's hardest,
  most-refuted area (DIS-6 offset/prefetch fixes all dead) — NOT churned further unsupervised.

## native LOW-T — engine floor (proven)
Full-kernel register-pinned asm (VAR_VIII) = +14.6% over LLVM (real, byte-exact, refutes
"asm can't beat LLVM") but plateaus at 0.667x ISA-L. Integration HELD (fails its own 0.85 gate;
a regression for isal which already has real ISA-L). => native pure-Rust no-C-FFI >=0.99-every-T
is very likely UNREACHABLE.

## BAR-1 REFRAME (post-T4-sizing): the isal blocker is T4 SCHEDULING, not T1
isal per cell: T1 has a KNOWN fix (single-shot, beats rg). T8 ties (0.990). T4 (0.911) is the
real wall — deeper parallel-scheduling/contention (consumer-lean refuted), the hardest area.
So >=0.99-EVERY-T for isal is gated at T4 by scheduling; T1 is recoverable either way.

## DECISIONS FOR THE USER (gate-flagged as goal-level, not supervisor-unilateral)
1. isal T1 FORK: (a) pragmatic — route T1/single-threaded isal to single-shot ISA-L (cheap,
   BEATS rg, but an isal-only path-fork off the "ONE PRODUCTION PATH / pure-Rust-sole" goal,
   does nothing for native); vs (b) faithful — lean the consumer (NOW SIZED: a TIE at T4, and
   at T1 it can't remove the inherent chunking-serialization tax => faithful lean leaves T1 ~0.90,
   misses 0.99). So pragmatic single-shot is the ONLY known path to >=0.99 at T1.
2. isal T4 SCHEDULING (the real BAR-1 blocker): the hardest, most-refuted area + high variance.
   Worth a fresh, user-steered attack (e.g. attack the 17-36% variance/contention directly), or
   accept ~0.91 at T4 — your call; I did NOT churn it unsupervised.
3. native BAR: 0.667x engine floor => accept native below bar at low-T / lean on isal as the
   at-parity build / revisit the >=0.99-every-T bar.

## OWED / READY (nothing running — chain stopped at the fork)
- Disasm machine-code confirmation (scripts/analysis/disasm_*): queued for box-free — the
  literal "is gzippy's linked ISA-L the AVX2 nasm kernel" check (source says SAME; owed empirical).
- JOB-2 reserve fix (gated-PASS, branch isal-resync-stored-fixed) + the build.rs comment fix
  (on HEAD): ready to merge.
- Several byte-transparent dev-oracle knobs left on worktrees (lean_consumer, singleshot,
  reserve-factor, marker/skip-writev) — flag to Steward; not merged.
- gzippy-NATIVE T4 0.761/T8 0.915 carry the same per-chunk pipeline + scheduling overhead ON TOP
  of the engine gap.

## LATE-NIGHT UPDATE (DIS-17): engine machine-code IDENTICAL; contention NULL; gap = +40% INSTRUCTIONS
- DISASM-PROVEN: gzippy + rg execute the byte-identical AVX2/BMI2 ISA-L `_04` nasm kernel (24-byte
  .text signature verbatim in both binaries). Engine-equivalence CLOSED.
- CONTENTION = clean NULL: frozen-box T4/T8 variance MATCHES rg (4%/4%, 10%/9% — the "17-36%" was
  thaw contamination); perf c2c false-sharing at noise floor; lock contention ~0; gzippy cache MPKI
  BETTER than rg. No contention.
- THE REAL GAP (frozen-box perf, T4): gzippy executes **+40% more instructions** (7.28e9 vs rg
  5.18e9) wrapping the IDENTICAL engine = the per-chunk ParallelSM pipeline doing more work/byte
  than rg's leaner consumer. Small wall-slack TLB-footprint term aside (DIS-14). => the BAR-1 lever
  is reducing the pipeline's +40% instruction count to rg's. NEXT: locate WHICH functions emit the
  extra ~2.1e9 instructions (perf annotate gzippy-pipeline vs rg) then faithfully converge.

## DIS-18 + GATE CORRECTION: the gap = per-byte MARKER-PIPELINE efficiency on a MATCHED fraction
- The +40% (= +1.54e9 user instr) is LOCATED: gzippy's pure-Rust u16-MARKER pipeline is ~57% of
  its instructions (read_internal_compressed, emit_backref_ring, push_slice, finalize, resolve).
- GATE REFUTED the "shrink the marker fraction via window-propagation" lever: vendor GzipChunk.hpp
  :661-710 shows rg ALSO marker-decodes speculative chunks (ISA-L u8-direct only under
  initialWindow && untilOffsetIsExact) => rg marker-decodes the SAME ~34.5% bytes (STEP-0 33.62 vs
  34.50; 73.0M vs 73.1M markers). So it is NOT a fraction gap. window-propagation also indep.
  refuted (DIS-6 can't deliver window pre-decode w/o serializing; OPEN-1 seed-all only via uncounted
  precompute).
- THE REAL GAP: gzippy's marker+scaffold pipeline runs ~1.5x rg's marker pipeline INSTRUCTIONS on
  the identical byte fraction (gz ~4.4e9 non-ISA-L vs rg ~2.8e9). PER-BYTE/structural efficiency.
- OWED (never done): perf-attribute RAPIDGZIP's marker decode (decodeChunkWithRapidgzip) apples-to-
  apples vs gzippy's marker engine -> splits the final lever: marker INNER LOOP (open inner-loop
  territory) vs SCAFFOLD/u16-width (faithful; width already wall-slack per page-warmth). RUNNING.

## DIS-19 + GATE: isal attribution COMPLETE. flip-to-clean is CONVERGENT (bombshell refuted).
- rg ALSO flips every streaming chunk's clean tail to separate ISA-L at 32KiB (GzipChunk.hpp:520-525
  finishDecodeChunkWithInexactOffset<IsalInflateWrapper> -> isal_inflate). gzippy's flip_to_clean is
  CONVERGENT — do NOT delete it (the owner's "rg never u8-direct" was a misread of a sibling symbol).
- FINAL SPLIT of the +1.54e9 isal excess (gated): ~71% marker INNER LOOP = ~1.70e9 u16 OUTPUT/BACKREF
  FRAGMENTATION (SegmentedU16 push_slice + separate u16-ring emit_backref_ring vs rg's ONE flat
  m_window16 inlined window[pos++] + single memcpy, deflate.hpp:926/1319/1376) [FAITHFUL lever, matches
  the one-MarkerRing memory] + ~1.61e9 pure-Rust Huffman vs ISA-L-as-primitive [asm, VAR_VIII-plateau,
  user-gated]; ~25% resolution scaffold; shared igzip kernel ~equal.
- THE FAITHFUL LEVER = flatten the fragmented u16 output toward rg's flat m_window16. BUT wall-payoff
  UNPROVEN (instr-count != wall; TIE-6 sized the u16-footprint axis wall-slack). OWED + RUNNING: a
  flat-buffer wall A/B removal oracle (does de-frag move the wall, or wall-slack like footprint?).
- BAR-1 read: the symbol-rate half (1.61e9) is asm-bounded (0.667x/0.900x plateau, gated); isal T4
  0.91->0.99 via the marker engine alone is unlikely. The de-frag is the one faithful candidate;
  its wall payoff decides whether it is worth the architectural port.

## DIS-20: de-frag is ON the T4 critical path (first non-refuted faithful lever) — port WARRANTED
- Slow-injection at the exact de-frag sites (emit_backref_ring + push_slice): T4 monotonic
  +243/+431/+772ms, freq-neutral SLEEP control survives (+177ms), byte-transparent. ON the T4
  critical path. T1 near-slack (+37ms, serialization-bound). Refutes the wall-slack null at T4.
- RULE-3 CAVEAT: this proves CRITICALITY, not the speed-up ceiling. The flat-buffer PORT's payoff
  (fraction of the ~60ms T4 gap recovered) is UNBOUNDED until the removal is actually built.
- DRIVING NOW: the faithful flat-buffer convergence (rg's flat m_window16 deflate.hpp:926/1319/1376
  replacing SegmentedU16 + u16-ring), byte-exact, measure the T4 wall SPEED-UP = the rule-3 removal
  bound + the real convergence. This is the campaign's one proven faithful lever; its wall result
  decides whether it closes isal T4 toward 0.99 or hits the next binder fast.
- (The other ~1.61e9 marker-Huffman-rate half stays asm-bounded/plateau, user-gated. native adds
  its 0.667x clean-path floor on top.)

## SUPERVISOR NOTE
Every owner number this session was supervisor-gated (owners can't spawn advisors here). One
phantom caught + corrected: a mislabeled-native-binary "ISA-L dormant" bombshell (reconciled).
The campaign's banked spine (scorecard, engine-same, the refutation chain) survived disproof.
