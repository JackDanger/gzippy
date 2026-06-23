# RANK-2 — instruction-surplus LOCATE (gz run_contig vs igzip _04), STEP-2 fix-list

**Date:** 2026-06-23  **Branch:** kernel-converge-A  **build sha = `a327982c`** (B2 f5f827b6
unconditional refill + B3 a327982c scalar overshoot-burst).  **Box:** Intel i7-13700T LXC
(guest REDACTED_IP), **UNFROZEN** (no_turbo=0; instruction count is deterministic unfrozen —
no freeze needed/used; bench-lock + llama untouched).  **Stamp: HYPOTHESIS-tier LOCATE —
single-arch Intel; the candidates below are RANKED by instruction-removal potential, NOT
proven wins. Each STEP-2 fix gates on cyc/wall FROZEN on BOTH arches (instr count is the
LOCATE metric, never the win verdict — B3 is the cautionary case: it removed a branch but
ADDED instructions). AMD/Zen2 owed.**
**Scope:** LOCATE + RANK only. NO code change this cycle.

## GATE-0 (all PASS)
- build sha == `a327982cf04551e3a067a2653c7e6ca7aa2ced3f` ✓
- build-flavor = `parallel-sm+pure`, path = `ParallelSM` (GZIPPY_DEBUG) ✓
- gz output sha256 == `zcat` == igzip output == `028bd0…cb410f` (211,968,000 B) ✓
- igzip self-test (2.31.1) byte-identical ✓
- run_contig IS the hot path: 87.1% of all gz retired instructions (perf instructions event,
  -c 50000); igzip decode_huffman_code_block_stateless_04 = 88.11% — both kernels ~88% in
  their inner loop, so kernel-vs-kernel is apples-to-apples ✓

## CONFIRMED PREMISE (deterministic, N=5 interleaved, `perf stat -x,`, taskset -c 4)
| metric | gz (-p1 forced SM) | igzip | surplus |
|---|---|---|---|
| whole-prog instr/B | **14.17** (14.163–14.178) | **11.40** (11.401–11.414) | **+2.76 (+24.2%)** |
| whole-prog cyc/B (UNFROZEN, context only) | 4.76 | 4.37 | +9% |
| whole-prog IPC | 2.98 | 2.61 | gz higher |
| **kernel-only instr/B** (share × whole) | **12.34** (87.1%×14.17) | **10.05** (88.1%×11.40) | **+2.29 (+22.8%)** |

The brief's "+2.0 instr/B (13.4 vs 11.4)" reproduces (~+2.3 i/B kernel here). The ~0.77 i/B
gap between gz whole (14.17) and gz kernel (12.34) is gz's parallel-SM DRIVER (CRC32, marker
resolution, apply_window, chunk lifecycle = the non-run_contig 10.3%) — a SEPARATE axis
(parallel scaffold), NOT this RANK-2 kernel locate.
**NUANCE (drives the cross-arch gate):** on Intel the +24% instr costs only +9% cyc because
gz runs at HIGHER IPC (2.98 vs 2.61 — the scalar burst is cheap independent stores). On Zen2
(per STEP-1 context, gz IPC ≥ igzip, instruction-bound) the same instr surplus converts more
directly to cycles ⇒ Zen2 is where instruction-removal pays. So the "cheap on Intel" scalar
burst may be the actual Zen2 cost; **the fix MUST be cyc/wall-gated on BOTH arches.**

## INSTRUCTION-GROUP ATTRIBUTION (perf instructions, address-bucketed to objdump regions)
gz instruction distribution (perf addr + 0x1000 = objdump addr; run_contig = c90d0–c95e0):

| region (objdump) | % of all gz instr | note |
|---|---|---|
| literal fast-loop  c9135–c91e1 | 41.2% | ~37 instr/iter; **≈ igzip's 38/iter — NOT the surplus** |
| **scalar 5-word burst copy  c9417–c94f8** | **19.6%** | **B3 path — the #1 surplus region** |
| dist-decode (94: in-reg entry)  c930c–c93a4 | 16.4% | mostly intrinsic (igzip decode_next_dist comparable) |
| non-run_contig (driver/CRC/marker/applywindow) | 10.3% | separate parallel-scaffold axis |
| rc prologue/long-table/bails | 7.6% | cold |
| B3 length-dispatch  c93a4–c93b3 | 1.7% | the cmp 0x28 / cmp 0xf0 router |
| **MOVDQU copy  c93b3–c9412** | **0.5%** | **≈unused** — silesia short backrefs bypass it |

### Why the literal path is NOT the lever (static instr-by-instr, near-cancel)
gz literal = 37 instr/iter, igzip loop_block literal ≈ 38 instr/iter. They differ by an even
trade: gz carries a **gz-only p0 un-consume anchor** (`lea p0,[pos*8]` + `sub p0,bitsleft`,
2 instr/iter, igzip is stateless) but igzip carries a **double EOB compare** (`cmp 256;je`
THEN `cmp 256;jl`, 2 extra instr) that gz fuses into one late discriminator (`cmp 0x100;jb`
hot, `je` only on fall-through). Net ~0. The +2.3 i/B is **the copy path**, not per-symbol decode.

## RANKED instruction-REMOVING fix candidates (STEP-2 list; each gated on cyc/wall FROZEN, BOTH arches)

### #1 — Copy path: route short backrefs through MOVDQU instead of the 40-byte scalar burst  ⭐ DOMINANT
- **Removes:** ~the 19.6% scalar-burst region. The B3 path (a327982c) routes ALL ≤40B
  backrefs to an UNCONDITIONAL 5-word (40-byte) scalar burst = ~10 mov + ~6 setup instr,
  writing 40 B for a mean-6.3-B copy. igzip copies the SAME backref with ~1 MOVDQU + sub16/jle
  (~4 instr, 16 B). gz HAS this MOVDQU path (c93b3) but B3 sends silesia's short backrefs
  away from it (MOVDQU only 0.5% vs scalar 19.6% = ~40:1 — silesia backrefs are nearly all
  ≤40B and dist≥8 → the heavy burst).
- **Vendor technique:** igzip `large_byte_copy` `igzip_decode_block_stateless.asm:603-612`
  (one MOVDQU per 16 B, src+=16) + `small_byte_copy` 614-627 (period-grow for overlap).
- **Cyc mechanism:** Intel — fewer retired stores (mem-port pressure ↓), but B3 burst is
  branchless so Intel cyc gain may be small (high-IPC absorbs it). Zen2 — direct retiring-bound
  win (this is the instruction-bound arch). **Net Intel/Zen2 split is exactly why it gates on both.**
- **Byte-exact feasibility:** MOVDQU path is ALREADY proven equivalent to emit_backref_contig
  for every dist/overlap (it's the production 41–240B path); a back-ref copy touches NO
  bit-cursor (bitbuf/bitsleft/pos) so c2/c3 cursor differential + ref model are unaffected.
  Hazard: the MOVDQU 16-B overshoot envelope (≤15 B) vs the scalar burst's 40-B extent —
  shrinking the write extent is envelope-SAFE (FAST_OUT_SLOP=282 >> both).
- **⚠ JUDGEMENT CALL (route to advisor, do NOT decide here):** B3 was deliberately ADDED to
  remove the nasa-dominant ≤40B trip-count mispredict branch (B3 commit msg). Reverting/narrowing
  it for silesia could REGRESS nasa wall (the original B3 motivation). The fix is NOT a plain
  revert — candidate shapes to A/B: (a) lower the B3 threshold so only the truly-tiny
  variable-trip case takes scalar and ≤40B-but-≥16B goes MOVDQU; (b) shrink the scalar burst
  from 5-word(40B) to 2-word(16B); (c) make the scalar-vs-MOVDQU split corpus-adaptive. Needs
  silesia AND nasa cyc/wall on BOTH arches before any cut.

### #2 — dist-decode inline validity/anchor tightening (16.4% region; LOW removable fraction)
- **Removes:** a few instr/backref. gz's in-register dist entry decode (94:) carries the
  resumable `saved_bitbuf` copy + inline validity branches (raw==0/je, dist==0/je,
  >window/ja, marker src<out_base/jb). igzip's decode_next_dist (`:396-440`) is comparable
  length and ALSO checks copy_start<start_out (`:583-584`), so MOST of this is INTRINSIC, not
  surplus. Only the gz-specific anchor maintenance (vs igzip stateless) is removable, and it
  is small here.
- **Cyc mechanism:** marginal; the branches are predicted-not-taken (retire but ~free on Intel).
- **Feasibility:** HIGH hazard / LOW reward — the validity checks are correctness-load-bearing
  (marker/out-of-window are the resumable bail contract). Low priority.

### #3 — literal-path p0 un-consume anchor removal (~0.27 i/B; 2 instr/iter, gz-only)
- **Removes:** `lea p0,[pos*8]` + `sub p0,bitsleft` on EVERY literal iteration (igzip stateless
  has no counterpart). ~2 of 37 literal instr ⇒ ~0.27 i/B.
- **Vendor technique:** igzip never un-consumes (stateless single-shot). gz needs the anchor
  for the resumable bail re-read at 85:.
- **Cyc mechanism:** removes a 2-op off-critical-path computation; small.
- **Byte-exact feasibility:** HARD — p0 is the bail un-consume bit-anchor; the NIGHT40 ledger
  already hoisted d0 off the literal path but kept p0 because reconstructing `bc` at a bail is
  the NIGHT32 register-live-range hazard (cyc/B regress). Removable ONLY via a structural
  un-consume redesign (e.g. igzip-style re-decode-from-scratch on bail), which is a larger
  convergence step, not an instruction tweak. Defer.

### #4 — B2 unconditional pre-copy refill (8-instr refill on the backref path)
- B2 (f5f827b6) made the pre-copy refill UNCONDITIONAL (deleted the `cmp bitsleft,48;jae 51f`
  skip) — instruction-ADDING when bitsleft≥48, but it removed a 36%-mispredict silesia branch
  and was KEPT cross-arch. Re-examining it is LOW priority (already justified; reverting
  re-introduces the mispredict). Listed for completeness, not recommended.

## RANKING SUMMARY
1. **Copy path — narrow B3 / MOVDQU short backrefs** (≈19.6% region, igzip 603-612) — by far
   the largest instruction-removal lever; **JUDGEMENT CALL (nasa-regression risk) → advisor.**
2. dist-decode tightening — mostly intrinsic, low reward.
3. literal p0 anchor — small, high hazard, needs structural redesign.
4. B2 refill — already justified, not recommended.

## REPRODUCE
```
# build native @a327982c on guest (RUSTFLAGS=-C target-cpu=native, --no-default-features
#   --features pure-rust-inflate, CARGO_TARGET_DIR=/dev/shm/rank2-target)
ssh -J REDACTED_IP root@REDACTED_IP
B=/dev/shm/rank2-target/release/gzippy; C=/root/silesia.gz
# instr/B (CSV, N=5 interleaved):
GZIPPY_FORCE_PARALLEL_SM=1 perf stat -x, -e instructions,cycles -- taskset -c 4 $B -d -c -p 1 $C >/dev/null
perf stat -x, -e instructions,cycles -- taskset -c 4 igzip -d -c $C >/dev/null
# region attribution:
GZIPPY_FORCE_PARALLEL_SM=1 perf record -e instructions -c 50000 -- taskset -c 4 $B -d -c -p 1 $C >/dev/null
perf report --stdio -F overhead,symbol   # bucket by objdump addr (perf+0x1000); run_contig=c90d0-c95e0
```

## cursor-agent locate-design REVIEW (incorporated)
cyc/byte is the FAIR primary fix metric, NOT instr-count (instr is the LOCATE/ranking signal
only — matches our bias guard); isolate the KERNEL SYMBOL, not whole-program (require ≥85%
cycles in run_contig — measured 87.1% ✓); use `perf record cycles:pp -c 5000` + `perf annotate`
for sub-region (cycle-skid on tiny asm loops makes default sampling ±several instr — trust for
hot-spot RANKING not single-instruction claims; corroborate with co-moving stall counters);
process-level TopdownL1/L2 for bucket diagnosis; sink-law (both >/dev/null), exclude CRC/table-
build/page-faults from the window, steady-state passes only. Pre-register fix-class per topdown
signature. Deliverable of a locate = median cyc/byte + topdown bucket + annotate-ranked
sub-region — NOT a rewrite recommendation (honored: this is a RANKED candidate list, gated next cycle).
