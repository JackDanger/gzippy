# ZEN2 WINDOW-ABSENT DECODE MICROBENCH — RESULTS + VERDICT

Branch `kernel-converge-A` @ cd157de2 (instrument) / measured on solvency AMD EPYC
7282 Zen2 (`root@10.0.2.240`). Pre-registration: `ZEN2-DECODE-MICROBENCH-FALSIFIER.md`.

## VERDICT: **REFUTE** — do NOT build the marker-decode asm.

gzippy's PRODUCTION window-absent (marker) decode loop is NOT slower per byte than
rapidgzip's — it is **FASTER** on all three losing cells. R = gz/rg cyc/B = 0.838
(median), well below the REFUTE threshold (≤1.10) and the OPPOSITE direction of
CONFIRM (≥1.20). The AMD ~3-6% rapidgzip residual is therefore NOT in marker-decode
speed; it is elsewhere (resolve / u16 traffic / apply-window / pipeline / process
lifecycle).

## SYMMETRIC cyc/B (decode-loop-only, denominator = marker-mode-emitted-bytes)
Frozen (gov=performance, boost=0), llama SIGSTOP'd for the window, taskset cores 0-3
(SMT siblings 16-19 idle), interleaved N=9 (gz, rg, gzAA per rep).

| corpus (T4) | gz cyc/B | rg cyc/B | R=gz/rg | A/A% | gz_wa_MB | rg_wa_MB | verdict |
|-------------|----------|----------|---------|------|----------|----------|---------|
| monorepo    |  8.333   |  9.529   | 0.874   | 1.9% | 38.4     | 41.4     | REFUTE  |
| silesia     | 10.941   | 13.062   | 0.838   |12.0% | 47.9     | 74.2     | REFUTE  |
| squishy     |  7.916   | 10.481   | 0.755   | 8.6% | 81.7     | 129.1    | REFUTE  |

gz is 13-25% FASTER per marker-mode byte. The silesia A/A spread (12%) is the only
loose cell, but the gz advantage (16% faster) is in the safe direction — even with
that noise gz is at worst at parity, never the ≥20%-slower a CONFIRM needs.

## THE DENOMINATOR ARTIFACT (why the prior "11.7 vs 3.85" CONFIRM was wrong)
The prior 3x gap compared gz's marker-loop cycles / its ~35% MARKER bytes (74.9 MB)
against rg's Block::read cycles / ALL output (212 MB) — mismatched denominators.
Measured with the SAME denominator (marker-mode-emitted-bytes) on both:
- rg true marker-decode cyc/B ≈ **13** (silesia), not 3.85. (986M cyc / 74.2 MB = 13.3
  in the contended Gate-0 single-shot; 13.06 frozen-interleaved.) The "3.85" was
  986M / 212MB(total) — a wall-contribution number, not the loop's per-byte cost.
- gz marker-decode cyc/B ≈ **10-11** (silesia) — comparable-to-faster than rg.

Secondary finding: gz spends LESS of its decode in marker mode than rg (silesia 47.9
vs 74.2 MB marker-mode bytes) — gz flips to the clean asm kernel sooner. Both the
lower marker-mode fraction AND the lower per-byte cost favor gz on the window-absent
path.

## METHOD (cursor-agent design-reviewed — symmetric production counters)
- gz: `GZIPPY_MFAST_PROF=1` + new MFAST_BYTES/CAREFUL_BYTES; cyc/B =
  (MFAST_CYC+CAREFUL_CYC)/(MFAST_BYTES+CAREFUL_BYTES). rdtsc starts AFTER per-block
  table build → decode-loop-only.
- rg: `RAPIDGZIP_WA_PROF=1` rdtsc guard around `readInternalCompressedMultiCached`
  gated `if constexpr(containsMarkerBytes)` (u16 window); cyc/B = cyc/nBytesRead.
  Patch: `scripts/bench/rg_wa_prof_patch.py`; backup `deflate.hpp.zen2bak`.
- Gate-0: both non-inert (gz counters fire on marker path; rg calls=1576 bytes>0),
  both OFF==identity, both sha(out)==zcat, A/A reported. Gate-4: gz path=ParallelSM.
- Driver: `scripts/bench/zen2_decode_microbench.sh`; report:
  `scripts/bench/zen2_decode_microbench_report.py`. CSV /dev/shm/zen2-mb-out/raw.csv.

## CURSOR-AGENT REVIEW (key critiques, all addressed)
1. Denominator must be marker-mode-emitted-bytes, not total → adopted (the crux).
2. Asymmetric isolated-rdtsc-vs-perf-annotate → replaced with symmetric in-binary
   counters on both (patched rg).
3. perf-annotate self-cycle attribution fragile under inlining → avoided (counters).
4. One-looped-block not representative → measure on full production runs.
5. rdtsc-under-contention not "frequency robust" → froze box + paused llama.
6. table-build asymmetry → gz timer starts after table build, rg builds in readHeader.

## SCOPE / TIER
STRONG (gated, symmetric counters, A/A-validated, N=9, three corpora) but AMD/Zen2
single-arch — Intel replication owed for LAW. Verdict holds at sha cd157de2 on Zen2.
Re-open trigger: a gated wall removal-oracle of the marker-decode region exceeding the
A/A spread, or an Intel result diverging.

## NEXT (where the AMD residual likely IS — NOT measured here, HYPOTHESIS)
The decode loop is acquitted. Candidates per prior banked work (all unvalidated):
apply-window / marker-resolve pass + u16 traffic (project_zen2_ceiling_u16: u16−u8
delta heavy), process-lifecycle teardown / peak RSS 1.65x (project_amd_t2_phase).
Those need their own gated discriminators.

## BOX HYGIENE — RESTORED + VERIFIED
llama-server 811408 state Rl; gov=ondemand; boost=1; watchdog killed (no orphan);
bench-lock.sh untouched. rg source /root/rg-build-src patched (env-gated identity when
RAPIDGZIP_WA_PROF unset); backup at deflate.hpp.zen2bak.
