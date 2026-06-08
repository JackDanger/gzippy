# Output-binder reconciliation — matched comparator (owner turn, HEAD 20084c91)

## The arithmetic tension the prompt asked to reconcile
- Output-REMOVAL oracle (GZIPPY_SKIP_WRITEV_SYSCALL) showed gzippy_skip ≈ rg (file sink) ⇒ "the whole ~34ms output is the gap, removing it ≈ parity."
- Yet the faithful overlap writer captured only ~6ms, with ~14ms attributed to a "shared memory-bandwidth floor."
- These don't square. Asked: of gzippy's ~35ms serial output, how much is (a) HIDEABLE under decode-WAIT (gzippy-specific recoverable) vs (b) SHARED with rg / irreducible (measure rg's output exposure too).

## What ran THIS turn (locked guest 10.30.0.199, taskset 0,2,4,6,8,10,12,14, gov=perf, interleaved best-of-N, silesia.gz RAW=211968000, gzippy native sha 028bd002…cb410f OFF byte-exact vs rg both at the start)
All numbers from interleaved batches (contention-matched). Sinks: `file` = fresh mktemp regular tmpfs file (= measure.sh's sink); `null` = /dev/null.

### Battery 1 (file-sink, sha-verified via measure.sh, N=13) — skip diverges BY DESIGN (wrong bytes), timings valid
- gz_off (file) 0.1692s = 0.767× rg
- gz_skip (file, writev removed) 0.1320s = 0.983× rg
- gz_overlap (file) 0.1602s = 0.810× rg
- rg (file) 0.1297s

### Battery 2 — MATCHED /dev/null comparator (N=13), the decisive disproof of the prompt premise
- gz /dev/null 0.1341s ; gz_skip /dev/null 0.1307s ; rg /dev/null 0.1156s
- ⇒ gz_null / rg_null = **1.160× — a ~18.5ms gap that SURVIVES output neutralization on BOTH sides.**
- gz_skip_dn (0.1307) ≈ gz_dn (0.1341): at /dev/null the writev is already nearly free (kernel discards), so skip barely helps — /dev/null already removes most output cost.

### Battery 3 — full interleaved split, both tools file vs null (N=15)
- gz_file 0.1677 ; gz_null 0.1324 ; gz_skip(null) 0.1318 ; gz_ovl(file) 0.1600 ; rg_file 0.1305 ; rg_null 0.1148

### Battery 4 — tight overlap (N=21)
- gz_file 0.1649 ; gz_ovl 0.1596 (+3.3%) ; gz_skip(null) 0.1311 ; rg_file 0.1301

## THE RECONCILIATION (numbers from Battery 3, the matched contention-paired one)
**Output exposure (file − null), per tool:**
- gzippy: 0.1677 − 0.1324 = **35.3ms** (gzippy's serial output exposure)
- rapidgzip: 0.1305 − 0.1148 = **15.7ms** (rg ALSO pays a serial output exposure)

**Split of gzippy's 35.3ms output exposure:**
- ~15.7ms = SHARED memory-bandwidth/page-cache floor (rg pays the same; both materialize 211 MiB serially on their in-order consumer) ⇒ NOT a gzippy deficit.
- ~19.6ms = gzippy-SPECIFIC excess (the dearer copy; advisor-D thesis = the u16-ring → u8 second pass leaves the consumer source buffer cold/double-touched).

**The SECOND binder that survives output removal on BOTH sides (matched /dev/null):**
- gz_null − rg_null = 0.1324 − 0.1148 = **17.6ms = engine+sched residual.** NOT addressable by output-overlap at all.

## CONSEQUENCE — the prompt's headline premise is REFUTED
"Removing gzippy's output ≈ parity" is an APPLES-TO-ORANGES pairing: it compares gzippy-output-removed (0.131) against rg-output-PRESENT (0.130). The matched comparator (both output-removed) leaves gzippy **1.16× behind**. There are TWO T8 binders, not one:
1. Output excess ~19.6ms (gzippy-specific, the overlap-writer lever's target; shared-floor 15.7ms is irreducible & rg pays it).
2. Engine+sched residual ~17.6ms (survives output removal both sides; the overlap lever CANNOT touch it).

## THE MAXIMAL OUTPUT-OVERLAP CEILING (the real ceiling for THIS lever)
Best case = drive gzippy's output exposure from 35.3ms down to rg's 15.7ms shared floor (perfect overlap, zero gzippy-specific excess). That lands gzippy at ≈ gz_null + 15.7ms ≈ 0.1324 + 0.0157 = **~0.148s = ~0.88× rg (file vs file).** The output-overlap lever's CEILING is ~0.88×, NOT parity — because Binder 2 (the 17.6ms engine residual) remains. The current overlap writer captures ~5-8ms of the ~19.6ms recoverable; ~12-14ms of recoverable headroom remains for a deeper/earlier overlap, but even fully captured it tops out at ~0.88×.

## RELATED ARCHITECTURAL FINDING (chunk_buffer_pool.rs:73-81, source)
gzippy spends ~40% CPU in asm_exc_page_fault + clear_page_erms vs rg's ~17% — std::alloc munmaps large-Vec frees and re-faults on next alloc; rg's rpmalloc per-thread arena keeps pages warm. This cold-page tax plausibly feeds BOTH Binder 2 (engine+sched residual) AND part of the output excess (cold pages in the writev gather). It is a separate, larger lever than output-overlap.

## CLAIMS TO DISPROVE (for the advisor)
- C1: The matched /dev/null comparator is the correct parity gate (not skip-vs-rg-file), and it shows ~1.16× ⇒ two binders.
- C2: The output-overlap lever ceiling is ~0.88× rg (output exposure can at best drop to rg's 15.7ms shared floor; the 17.6ms engine residual is untouchable by this lever).
- C3: rg's 15.7ms output exposure is the genuine SHARED floor (both tools serially materialize 211 MiB; rg overlaps per-chunk and still pays it) ⇒ ~15.7ms of gzippy's 35.3ms is NOT a gzippy deficit.
- C4: The next-larger lever is NOT output-overlap (capped at ~0.88×) but the engine+sched residual / cold-page-fault tax (the unified-decoder + arena thesis).
