# Arch-dispatch the selector constants + locate the Zen2 T3 residual (2026-06-27)

Branch `feat/silesia-t3-output-crossover` @ `60048427` (off `b0e01b4a`).
Closes the cross-arch concurrency win: makes the serial-clean floor selector
LAW-grade on BOTH Intel (Raptor Lake) and AMD (Zen2) via runtime CPU-vendor
dispatch of its constants, and LOCATES the remaining Zen2 T3 residual.

## Scope stamp / instruments

- **AMD**: EPYC 7282 (Zen2, 16C/32T), `root@REDACTED_IP`, **RUNNING THE USER'S
  LLAMA** the whole session — never killed/paused/frozen; `llama-server` STAT
  verified `Rl` before/after every cycle. Build: `/root/gz-xarch2` @ 60048427,
  `RUSTFLAGS=-C target-cpu=native --no-default-features --features gzippy-native`,
  `/dev/shm/archdisp-target`, `build-flavor=parallel-sm+pure`,
  `GZIPPY_DEBUG=1 → path=ParallelSM`, all corpora sha-verified vs `gzip -dc`.
- **Intel**: i7-13700T (Raptor Lake, 16 logical), `ssh -J neurotic root@REDACTED_IP`,
  `/root/gz-floor` @ 60048427, `/dev/shm/sel-target`, same flavor/path, sha OK.
- **WALL** (load-immune): `scripts/bench/_archdisp_wall.py` — every arm
  (gz_on=selector default, gz_off=`MARGIN=0`=parallel-always, rg, igzip) run
  back-to-back per rep (interleaved → llama drift hits all arms equally), median
  of N≥9 per-rep PAIRED ratios, `/dev/null` sink, sha-verified, + an A/A
  (gz-on-twice) noise-floor arm (sat at 0.97–1.04 → ±2-3% noise floor).
- **ATTRIBUTION** (load-immune): `fulcrum abmeasure` (cyc/B + instr/B,
  interleaved, perf-stat, sha-verified) parsed by `scripts/bench/_abparse.py`.
  instr/B is the most contention-immune signal (retired-instruction count of the
  decode process is invariant to background load); cyc/B is gated against the A/A.
- AMD absolutes are HYPOTHESIS-grade (VOID under llama); only the paired ratios
  are verdict-bearing.

## JOB 1 — ARCH-DISPATCH (clean gated fix, byte-identical)

`effective_parallel_threads` (`src/decompress/parallel/single_member.rs`) now
selects its two constants by runtime CPU vendor (`cpuid` leaf 0, cached;
`AuthenticAMD` → Zen tune, else Raptor-Lake tune; non-x86_64 → Raptor-Lake):

| constant                 | Intel / other | AMD / Zen2 |
|--------------------------|---------------|------------|
| crossover margin         | 1.0           | **1.6**    |
| large-output bonus depth | 1 notch       | **2 notch**|
| large-output threshold   | 128 MiB       | 128 MiB    |

Env knobs remain overrides: `GZIPPY_PARALLEL_CROSSOVER_MARGIN`,
`GZIPPY_PARALLEL_LARGE_OUTPUT_NOTCH`, `GZIPPY_PARALLEL_LARGE_OUTPUT_BYTES`.
Byte-identical (the selector only changes the parallel THREAD-COUNT; output sha
== gzip on both arches, every corpus). 955 lib tests + clippy clean; existing
selector tests pinned to explicit constants so they are arch-independent, +3 new
tests (arch defaults match vendor; env override wins; 2-notch behaviour).

### Why margin 1.6 + 2-notch (not the prior "margin 1.6 + bonus OFF")

The XARCH plan's two fixes were each gated in ISOLATION at margin 1.0; their
COMBINED effect at margin 1.6 was never measured. Measured here (Zen2, N≥9):
margin 1.6 + bonus-OFF over-inflates the LARGE-output crossovers and routes
**silesia-T3 to SERIAL → loses rg 1.061** (regresses the north-star cell) and
squishy-T3 to serial (1.279). Root cause: Zen2 needs margin 1.6 only for SMALL
outputs (monorepo, where parallel never beats serial through T8); for LARGE
outputs 1.6 pushes the marginal-parallelism knee one notch too high. A **second**
bonus notch lands silesia at crossover 3 (T3 parallel) and squishy at 2 (T2
parallel), restoring/improving both. The output-size discriminator is exactly
what separates the small-output (monorepo) case from the large (silesia/squishy).

### GATED — AMD/Zen2 all-T curve, DEFAULT (arch-dispatched) constants, N=11

```
corpus    T   on/off  on/rg  on/igz  aa   note
silesia   1   0.997   0.768  0.956  0.997
silesia   2   0.838   0.789  0.954  0.998 serial (parallel-T2 would regress)
silesia   3   1.002   0.975  0.874  1.005 PARALLEL — BEATS rg (was tie 0.992)
silesia   4   0.997   0.925  0.680  1.005
silesia   6   0.980   0.900  0.580  0.969
silesia   8   1.007   0.846  0.516  0.992
silesia   16  1.011   0.878  0.456  1.042
monorepo  1   0.998   0.470  0.952  0.998
monorepo  2   0.443   0.374  0.960  0.998
monorepo  4   0.672   0.494  0.957  1.000
monorepo  6   0.778   0.563  0.962  0.995 serial — BEATS igzip (was 1.254 REGRESSION)
monorepo  8   0.939   0.603  0.967  1.007 serial — BEATS igzip (was 1.043)
monorepo  16  1.015   0.642  0.931  0.991
squishy   1   1.003   0.792  0.967  1.000
squishy   2   0.998   0.927  0.987  0.998 BEATS igzip+rg (was 1.013 REGRESSION)
squishy   3   0.997   1.008  0.771  0.997 TIE vs rg (was 1.031 loss) — see Job 2
squishy   4   0.999   0.970  0.607  1.005
nasa      1-16 ~1.00  0.28-0.74 0.90-0.92 ~1.0 flat (hard-capped serial)
```

**AMD verdict: gz DOMINANT vs igzip at EVERY cell; vs rg at EVERY cell EXCEPT
squishy-T3 (1.008 = tie within A/A 0.997).** The three default-constant
regressions (monorepo-T6/T8, squishy-T2) are ERASED; silesia-T3 flipped
tie→BEAT and squishy-T3 loss→tie. No cell regressed vs the pre-fix baseline.

### GATED — Intel/Raptor-Lake no-regression, DEFAULT constants, N=11

cpu_is_amd()=false on Intel ⇒ constants stay margin 1.0 + 1-notch ⇒ routing is
byte-identical to b0e01b4a. Confirmed at the wall:

```
silesia : T3 on/rg 0.853 (BEATS rg, parallel) ; all T beat rg 0.79-0.85, monotonic
monorepo: all T beat rg 0.41-0.66 + igzip 0.77-0.96 ; T6 0.964 (stays parallel — OK on Intel)
squishy : T2 on/rg 0.853 on/igz 0.913 (WIN) ; T3 0.885 ; T4 0.886 — all beat rg+igzip
```

No Intel regression — the Raptor-Lake curve is unchanged and still dominant
(silesia-T3 still beats rg, the BEAT-plan win is intact). **Floor is now
cross-arch LAW-grade with arch-dispatched constants.**

## JOB 2 — LOCATE the Zen2 T3 residual (kernel floor vs recoverable overhead)

The only sub-parity Zen2 cell after Job 1 is **squishy-T3 (1.008, a tie)** —
silesia-T3 now BEATS rg (0.975). `fulcrum abmeasure` decomposition (load-immune,
N=7-9, A/A-gated, rg arm sha-verified independently = `gzip -dc`):

```
squishy gz-PARALLEL-T3 vs rg-T3:   gz  cyc/B 10.84  instr/B 23.54  IPC 2.17
                                   rg  cyc/B 10.50  instr/B 23.56  IPC 2.24
   gz/rg instr/B = 0.9998   gz/rg cyc/B = 1.0349   A/A cyc/B 1.008
squishy gz-SERIAL(-p1)  vs rg-T1:  gz  cyc/B  5.30  instr/B 14.01  IPC 2.64
                                   rg  cyc/B  7.15  instr/B 14.69  IPC 2.06
   gz/rg cyc/B = 0.754  (gz serial is 25% FEWER cyc/B than rg-T1)
silesia gz-PARALLEL-T3 vs rg-T3:   gz instr/B 22.99  rg 23.34  → gz/rg instr/B 0.986
   (gz does FEWER instructions → explains the silesia-T3 wall BEAT 0.975)
```

### VERDICT: it is the per-symbol DECODE KERNEL (a), NOT recoverable parallel-overhead (b).

- The parallel scaffold (marker-resolution + apply_window + per-chunk) costs gz
  **+9.5 instr/B** over serial (14.0 → 23.5 instr/B — the W-inflation the selector
  models). **But rg pays the IDENTICAL 23.56 instr/B** — rg carries the same u16
  marker machinery + apply-window pass (CLAUDE.md: rg 31.25% replaced-marker
  symbols, 0.113s apply-window). So **gz and rg do EQUAL total parallel-path work
  at squishy-T3 (instr/B ratio 0.9998)** — there is NO gz-specific excess
  scaffold to shave, hence **no cheap recoverable parallel-overhead lever**.
- The entire squishy-T3 residual is the **3.5% cyc/IPC gap** (gz IPC 2.17 vs rg
  2.24) on the per-symbol decode kernel — the documented Zen2 pure-Rust-vs-ISA-L
  codegen floor (`project_amd_rworker_subdecomp`: intrinsic packaging floor,
  recoverable budget ≈ 0). gz's SERIAL kernel is actually 25% FEWER cyc/B than
  rg-T1; the gap appears only under the parallel scaffold's lower-IPC regime, and
  it is codegen, not extra work.
- silesia-T3 BEATS rg precisely because there gz emits FEWER instructions per
  byte (0.986) — same kernel domain, corpus-favourable. The tie-vs-beat split
  between squishy and silesia is corpus-dependent kernel efficiency, both small.

**Can squishy-T3 be flipped tie→beat on Zen2 short of the big asm project? No.**
The recoverable parallel-overhead budget is ≈ 0 (gz and rg pay equal scaffold).
Closing the residual 3.5% requires ISA-L-grade Zen2 codegen for the per-symbol
decode kernel (close the IPC gap) — the large structural project, NOT a routing
or cheap-scaffold lever. squishy-T3 at 1.008 is already a TIE (within the A/A
0.997 noise floor) and beats igzip 0.771; silesia-T3 (the user's named cell) is
BEATEN on Zen2 (0.975). HYPOTHESIS (unvalidated): a Zen2-specific kernel codegen
pass would flip squishy-T3 to a beat — only a built-and-measured kernel change
can confirm; pre-registered falsifier = no IPC improvement on the squishy-T3
abmeasure.

## Status of claims
GATED (this session): Job-1 byte-exactness + 955 tests + clippy; the AMD all-T
default-constant curve (regression erasure, silesia-T3 beat, squishy-T3 tie,
N=11 load-immune interleaved paired, A/A 0.97-1.04, sha-verified); Intel
no-regression spot curve (N=11); Job-2 squishy-T3 instr/B=1.00 + cyc/B=1.035
attribution (A/A-gated) ⇒ kernel-floor not recoverable-overhead. The user's bar
"beat silesia-T3 everywhere" is MET (Intel 0.853, Zen2 0.975). Reproduce (AMD):
`scripts/bench/_archdisp_wall.py --bin <native> --rg <rapidgzip> --igzip igzip
--corpus <f.gz> --tlist "1 2 3 4 6 8 16" --cores 0-31 --n 11`.
```
