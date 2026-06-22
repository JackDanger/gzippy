# AMD/Zen2 rapidgzip-residual re-attribution — DESIGN (for cursor-agent review)

Branch kernel-converge-A @ HEAD 43923e84. Box solvency (AMD EPYC 7282 Zen2, root@REDACTED_IP).

## CONTEXT (what is already gated)
- DECODE KERNEL ACQUITTED (commit 43923e84, symmetric counters, cursor-agent-reviewed):
  gz window-absent marker decode is 13-25% FASTER per marker-mode byte than rg
  (R=0.838 silesia-T4). The prior "11.7 vs 3.85 cyc/B" was a DENOMINATOR MISMATCH
  (rg's 3.85 was cyc÷TOTAL-output, not cyc÷marker-mode-bytes; true rg ~13).
- CLEAN-WALL reconfirm: real ~3-6% gz>rg T2/T4 wall loss on silesia/monorepo/squishy
  on Zen2 (Intel = TIE). AMD-T4 framed "WORK-bound, gz ~+8% retired cycles".
- T2 phase instrument: T2 excess attributed to process-lifecycle TEARDOWN (peak RSS
  gz 97MB vs rg 59MB = 1.65x; process::exit unmaps full RSS), decode at rg-parity.

## QUESTION
The decode is acquitted; resolve was found cheaper than rg's (perf flat 7.80%).
So WHERE do gz's +~4-8% T4 cycles actually go? Re-attribute the FULL T4 cycle budget
gz-vs-rg across NON-decode regions with CORRECT region denominators and SYMMETRIC
in-binary counters (NOT asymmetric rdtsc-vs-perf-annotate — the prior denominator bug).

## UNITS (load/freq-robust, distpreload principle — NO freeze, NO llama pause)
In-binary `rdtsc` (TSC) accumulators on BOTH binaries. TSC is frequency-invariant, so
region cyc/B is load- and frequency-robust by construction. taskset-isolate cores 0-3
(llama roams elsewhere), interleave arms (gz,rg,gzAA), best-of-N>=9, GHz-stability gate
(reject if achieved GHz spread across arms > 1%). ZERO host mutations.

## REGIONS (symmetric; gz site -> rg site)
- R_dec_marker : marker-mode huffman decode
    gz MFAST_CYC+CAREFUL_CYC (have) ; rg RAPIDGZIP_WA_PROF (have)
    denom = marker-mode-emitted bytes (both)
- R_dec_clean  : clean huffman decode
    gz contig C_CYC_CALL (have) ; rg Block::read clean portion / isal loop_block (ADD)
    denom = clean-emitted bytes (both)
- R_resolve    : marker resolution / applyWindow
    gz resolve_chunk_markers_on_chunk (ADD rdtsc) ; rg ChunkData::applyWindow (ADD)
    denom = marker bytes resolved (both)
- R_crc        : crc32
    gz crc32 update sites (ADD) ; rg crc32 call (ADD)
    denom = total output bytes (both)
- R_output     : narrow u16->u8 + copy/write to sink
    gz narrow_u16_to_u8 + writev (ADD) ; rg append/write (ADD)
    denom = total output bytes (both)
- R_materialize: gz builds predecessor window from segments (materialize_window) (ADD)
    rg keeps windows in WindowMap — POSSIBLY gz-only (candidate excess). denom = chunks
- R_other      : perf_total_cyc - sum(above) = pipeline/consumer coordination/alloc/idle

## CONSERVATION (Gate-0e)
sum(regions) + R_other == total. Require R_other >= 0; report its fraction. A huge
R_other means the partition missed a region.

## ATTRIBUTION OF THE EXCESS
Dtotal = gz_total - rg_total (the +8%). Per region D_r = gz_r - rg_r.
Require sum(D_r) ~= Dtotal (conservation of the DIFFERENCE).
LOCATED lever = region with D_r >= 0.40*Dtotal AND gz_r > rg_r. Else "distributed/intrinsic".

## GATES
Gate-0: each counter non-inert (>0 both arms); OFF==identity (sha(out)==zcat both);
A/A (gzAA vs gz region cyc within spread); matched denominators; conservation reported.
Gate-1: interleaved best-of-N>=9, D vs inter-run spread.
Gate-4: path=ParallelSM, HEAD sha, gzippy-native build.
Gate-2 (verdict-strength, NEXT cycle only): removal-oracle on the located region.

## CONFOUNDS — cursor-agent please rule on these
1. TSC vs perf-cycles unit mismatch for conservation: TSC=invariant ref freq;
   perf `cycles`=actual core cycles. Without freeze they differ. Options: (a) freeze
   gov=performance+boost=0 so TSC~=core-cycles (box mutation, trap+watchdog restore,
   NO llama pause); (b) use a TSC-based total (process-level) instead of perf; (c)
   skip a perf-total and compute R_other as a per-thread rdtsc residual. Which is sound?
2. Cross-thread cycle summing: region counters are global atomics summed over worker+
   consumer threads. Is comparing thread-summed-TSC-work to a perf core-cycle total
   valid at T4? Does idle/spin contaminate the total?
3. Double-counting / nesting: decode rdtsc is inside the worker; a worker-total span
   that also contains crc would double-count crc. (Prior "62ms serial CRC phantom" was
   a nested-span double-count.) How to structure spans to avoid nesting double-count?
4. R_resolve denominator symmetry: gz resolves data_with_markers (u16>=MARKER_BASE);
   rg applyWindow resolves its u16 marker array. Same marker-byte population? Confirm.
5. R_dec_clean boundary: rg clean tail is ISA-L (loop_block); gz clean is the asm
   kernel. Wrapping rg Block::read vs gz contig — both "clean huffman decode" over
   clean-emitted bytes. Valid symmetric boundary?
6. R_materialize: is it truly gz-only (no rg counterpart)? If so the WHOLE region is
   gz-specific excess — but must confirm rg pays no equivalent window-assembly cost.
7. R_output narrow: gz narrows u16->u8 then writes; rg ALSO has u16 markers — does rg
   narrow too (symmetric) or write u8 directly (gz-only narrow)?
8. Is per-region cyc/B the right comparator, or absolute region cyc (since the excess
   is an absolute work delta)? Plan uses BOTH; is that double-billing in the verdict?

---
## CURSOR-AGENT REVIEW (2026-06-22) + ADOPTED REVISIONS
Verdict: revise before running. Symmetric-counter idea right; conservation model unsound as written; resolve/output/materialize need exclusive boundaries.

ADOPTED:
- ONE CURRENCY: thread-summed EXCLUSIVE TSC totals; compute R_other from the TSC total.
  Do NOT conserve rdtsc-regions against perf `cycles`. perf cycles only as a side-channel
  sanity for a pure gz-vs-rg TOTAL comparison (same metric both arms = valid), and to
  check perf/TSC ratio stable per arm.
- EXCLUSIVE LEAF spans only, no nesting. resolve_chunk_markers_on_chunk includes
  resolve+narrow (+subchunk window pop) — it is ONE leaf region "resolve+narrow"; do NOT
  also count narrow in R_output. CRC is worker-side (finish_decode commit loop), separate
  leaf, not nested in the decode spans.
- VERDICT criterion = D_r = gz_abs_cyc - rg_abs_cyc (ABSOLUTE, thread-summed TSC). cyc/B +
  bytes/counts are DIAGNOSTIC only (no deciding-by-both = no double-billing).
- BLOCKING spans (consumer dequeue wait) = wait, NOT work → keep them OUT of work regions
  (instrument the post_process+write CPU work, not the queue wait).
- R_materialize NOT assumed gz-only — add rg WindowMap/window-production counter even if ~0.
- R_resolve denom = bytes in marker-mode buffer (same population both); also report
  marker-symbol count as sanity.
- R_dec_clean: both counters must wrap the same clean-decode scope, denom clean-emitted bytes.

STAGED EXECUTION (bank each): S0 re-confirm premise (perf total cyc gz-vs-rg, existing
binaries) → S1 gz exclusive-TSC region partition → S2 rg symmetric patch → S3 absolute-delta
verdict. perf-cycles total for S0 is sound (same metric both arms).
