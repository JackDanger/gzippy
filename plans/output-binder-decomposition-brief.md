# Output-binder decomposition — fulcrum_total + removal oracle (owner turn, HEAD 864d12e9)

## What ran
1. **fulcrum_total** on the production gzippy-native T8 trace (byte-exact 028bd002…cb410f,
   ParallelSM, window_seeded=2 natural propagation — tool REFUSES production-certification on
   seeded>0 but the descriptive WAIT/COMPUTE/OUTPUT split is a genuine cross-check, busy+idle==span).
   CRITICAL: captured with the PRODUCTION sink. First capture piped to `sha256sum` (a pipe) inflated
   `consumer.writev` from 58→135ms (backpressure phantom). Re-captured to a regular tmpfs file
   (exactly what measure.sh's mktemp does) → the trustworthy split.
2. **writev-removal oracle** (NEW knob GZIPPY_SKIP_WRITEV_SYSCALL=1, byte-transparent OFF==identity,
   wrong-bytes ON, terminal trailer CRC still proves the full decode ran).
3. **/dev/null vs tmpfs-file sink** A/B (both tools, interleaved).
4. **writev-granularity sweep** (NEW knob GZIPPY_WRITEV_CAP_KIB, byte-exact all caps).
5. **T1 vs T8** removal oracle (Amdahl / contention discriminator).

## fulcrum_total decomposition (consumer = the wall-critical thread, production tmpfs-file sink)
Consumer span ~183-192ms (instrumentation-inflated vs locked-guest 165ms, but the SPLIT is the signal):
- **wait-on-workers (engine showing through): 64-67% (~123ms)**
- **output / consumer.writev: 32-34% (~58-65ms)**
- serial consumer compute: 0.8% (negligible)
Other SELF terms: worker.block_body 760-780ms SUM (parallel, slack-masked); post_process.apply_window
63-79ms; pool.pick.wait (idle worker wait).

## CAUSAL removal oracle — DECISIVE (locked guest, interleaved measure.sh, same tmpfs sink, N=13-15)
| contender | T8 wall | vs rg |
|-----------|---------|-------|
| gzippy_off | 0.1653s | 0.79× |
| **gzippy_skip (writev removed)** | **0.1310s** | **~1.00× (== rg 0.1311s) TIE** |
| rapidgzip | 0.1311s | 1.0 |

**Removing gzippy's output writev closes the ENTIRE remaining T8 parity gap.** The OUTPUT term is
the single largest binder at T8.

## Sink A/B (independent corroboration)
| sink | gzippy | rg | gzippy serial-output floor |
|------|--------|-----|------|
| tmpfs file | 0.1652 | 0.1276 | — |
| /dev/null | 0.1306 | 0.1138 | gzippy saves ~34ms, rg saves ~14ms |
⇒ gzippy's serial output floor (~34ms) is ~20ms more expensive than rg's (~14ms).

## The structural cause (Amdahl, NOT contention, NOT writev granularity)
- **writev granularity does NOT matter:** GZIPPY_WRITEV_CAP_KIB ∈ {2048,256,95} all TIE/worse vs off.
  So the cost is NOT the few-large-vs-many-small syscall shape (gzippy 17×12.5MiB writev vs rg
  2223×95KiB write). It is the intrinsic 211 MiB kernel page-cache COPY.
- **T1 discriminator:** at T1 gzippy_off 0.513s, skip 0.465s (+10%), rg 0.315s — at T1 the ENGINE
  binds (gzippy 0.615× rg; output removal does NOT close it). At T8 the per-thread engine work is
  ~8× parallelized but the serial 211 MiB output copy on the ONE in-order consumer does NOT
  parallelize ⇒ it becomes the dominant serial tail at high T (Amdahl). This is why output is the
  T8 binder but the engine is the T1 binder.

## REVISES the prior advisor-vetted split
Prior: "0.135× engine residual (table-load latency) + 0.075× non-engine (placement)". The
writev-removal oracle shows the T8 binder is OUTPUT (the serial 211 MiB materialization), and the
engine is largely OVERLAPPED/hidden at T8 (gzippy_skip == rg). The prior "table-load latency engine
residual" is the T1 binder, slack-masked at T8. The non-engine term is BIGGER than 0.075× and is the
serial output tail, not placement/scheduling.

## ADVISOR DISPROOF (synchronous, plans/output-binder-advisor-verdict.md)
- A UPHELD-W-CAVEATS: ≥14ms is the irreducible 211MiB page-cache copy floor (rg pays it); only the
  ~20ms EXPOSURE over rg is addressable. The removal oracle over-bounds the achievable win by ~14ms.
- B REFUTED-as-stated: "skip==rg" used an rg number (0.1276) from a different batch than the tie
  number; same-batch skip-vs-rg is 0.982× (skip ~1.8% behind rg), NOT a perfect tie. Output is still
  the dominant remaining binder but there's a ~2% engine/sched residual.
- E (the strongest disproof, "the 20ms is engine feed-rate masked, not output"): **REFUTED by the
  owed instant-feed discriminator** (below). 

## INSTANT-FEED DISCRIMINATOR (the advisor's owed oracle) — tmpfs file sink, same interleave, N=13
| feed | off (file) | skip (no writev) | OUTPUT EXPOSURE (off−skip) |
|------|-----------|------------------|----------------------------|
| unseeded (slow, prod engine) | 0.1669 | 0.1314 | **35.5ms** |
| seeded (fast feed, −38ms) | 0.1240 | 0.0933 | **30.7ms** |
| rapidgzip | 0.1290 | — | — |
Speeding the engine by 38ms (feed 0.131→0.093) changed output exposure only 35→31ms. If exposure
were feed-rate-gated it would have collapsed toward the ~14ms floor. It did NOT ⇒ **output exposure
is engine-INDEPENDENT, a genuine serial batched-write cost.** E's phantom is REFUTED. Also: seeded
gzippy 0.124s BEATS rg 0.129s WITH full output, and seed_skip 0.093s ≪ rg — gzippy's clean engine is
excellent; the prod gap is window-absent bootstrap (a known, separate term) + this output exposure.

## THE FAITHFUL BOUNDED TECHNIQUE (advisor-recommended, ceiling ~20ms = ~0.10× rg)
Mirror rg's writeFunctor (ParallelGzipReader.hpp:521): take the 211MiB serial batched writev OFF the
in-order consumer's critical path so it overlaps the next chunk's decode-WAIT (consumer is 64-67%
WAIT = ample slack to hide a copy whose floor is ~14ms). gzippy currently batches into 17×12.5MiB
writev that land EXPOSED on the serial consumer. Granularity-capping is REFUTED (size≠timing); the
load-bearing variable is write TIMING/overlap. Bounded by the writev-removal oracle (~35ms removed)
minus the irreducible ~14ms floor ⇒ achievable ceiling ~20ms ≈ 0.10× rg.
