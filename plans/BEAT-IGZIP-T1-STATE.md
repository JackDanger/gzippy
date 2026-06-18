# BEAT-IGZIP-T1 — DURABLE STATE

Mission: make gzippy-native (pure-Rust, FFI-off) **T1 single-member gzip DECODE**
measurably FASTER than igzip (ISA-L), byte-exact, gated. Single-arch Intel = NOT LAW
(AMD/Zen2 replication owed). T1 single-core only — no T4/T8 extrapolation.

Branch: `perf/igzip-full-rewrite`. Mac worktree (edit/commit/push only — aarch64,
CANNOT run the asm): `/home/user/www/gzippy/.claude/worktrees/agent-a8069a92d914fcef3`.
Guest (ONLY x86_64/BMI2 measure box): `ssh -J REDACTED_IP root@REDACTED_IP`.
Guest worktrees: gzippy(B)=/root/gz-fullrewrite (kernel 2b10aa48 dist-preload),
baseline=/root/gz-baseline (8383a2eb). igzip=/usr/bin/igzip (ISA-L 2.31.1).
Harness on guest: /root/distpreload-harness/.

## COMMITS THIS MISSION
- 2c135d07 — deliverable #0: commit orphaned paired harness (analyzer+memstress+driver) to scripts/bench.
- 2e01dd4f — deliverable #1: add igzip arm `scripts/bench/_gzippy_vs_igzip_paired_guest.sh`.

## THE INSTRUMENT (deliverable #1) — Gate-0 SELF-VALIDATED, PASS
`scripts/bench/_gzippy_vs_igzip_paired_guest.sh` (reuses committed `_distpreload_paired_analyze.py`).
arm A(A1,A2)=igzip, arm B=gzippy-native (ParallelSM @ T1). medΔ=(B-A1) cyc/byte;
NEGATIVE => gzippy faster. Gate-0 verified live: gzippy run_contig KERN entries>0
(monorepo 8299, nasa 25399), igzip --version printed, BOTH arms sha==zcat ref==each
other, same /dev/null sink + same pin(cpu4), GHz spread <0.07%, A2-A1 self-test ~0.
Run: `PIN=4 REPS=21 CORPORA="silesia monorepo nasa" /root/distpreload-harness/_gzippy_vs_igzip_paired_guest.sh`

## THE GAP TO CLOSE — gzippy-native T1 vs igzip cyc/byte (whole-process, /dev/null sink) — GATED
FULL N=21 paired, Wilcoxon p, bootstrap CI, A2-A1 self-test, + bandwidth stressor.
Intel i7-13700T LXC (cpu4 pinned). All cells p=6.4e-5, CI excludes 0, self-test PASS
(except stressed-nasa, noted). medΔ=(B-A1)>0 means gzippy SLOWER. THE GAP IS LARGE:

UNSTRESSED (clean) — THE number to close:
| corpus   | igzip cyc/B | gzippy cyc/B | medΔ (B-A1)      | Δinstr/byte | ΔIPC   |
|----------|-------------|--------------|------------------|-------------|--------|
| silesia  | 4.30        | 6.00         | +1.70 (+39.5%)   | +~2.3       | -0.27  |
| monorepo | 2.97        | 5.17         | +2.20 (+73.9%)   | +2.83       | -0.43  |
| nasa     | 1.58        | 3.43         | +1.84 (+116.5%)  | +2.08       | -0.64  |

STRESSED (memstress 14T, LLC-miss ~42-45%): gap HELD/GREW (silesia +39.5%, monorepo
+108%, nasa +179%) ⇒ NOT a bandwidth artifact; it is real work-volume. (stressed-nasa
self-test FAILED 2.9% — that ONE cell's precision is degraded; effect dwarfs it; use
the unstressed nasa cell as the clean number.)

GATED FACTS (this commit, Intel-only, NOT-YET-LAW — AMD owed):
1. gzippy-native T1 is **40-116% SLOWER than igzip** (cyc/byte), gated, 3 corpora.
2. The gap is **INSTRUCTION-DOMINATED**: gzippy retires **+2.0 to +2.8 more instr/byte**
   than igzip (ΔIPC also negative). This is WORK VOLUME, not micro-latency — a preload
   tweak will NOT close it; the lever is REDUCING instr/byte in the hot loop.
3. **run_contig = 85.36% of T1 self-time** (perf, symboled binary, silesia). Scaffold
   (block finder / marker machinery / apply_window / CRC) is only ~15%. So the excess
   instructions ARE in run_contig — the mission's kernel focus is correctly aimed.
4. Hot per-symbol economy in run_contig (perf annotate, top self-time instrs):
   multiple classify branches PER SYMBOL — `cmp $0x100`(EOB) 5.8%, `cmp $0xff`(literal-
   range) 3.6%, `test $0x2000000`(marker/reclass discriminator) 2.8%, plus two table
   loads `mov (%r11,%rXX,4)` (6.3%+) and the engaged MOVDQU backref copy (4.5%x2,
   8383a2eb is LIVE). igzip's loop_block collapses this to ONE speculative store + ONE
   discriminator branch — THAT delta is the instruction excess to attack.

## NEXT (planned, in priority order) — for the iteration phase (deliverable #2)
The target is now precise: cut instr/byte in run_contig toward igzip's loop_block
economy (fewer classify branches per symbol; speculative unconditional store; single
discriminator branch). Candidate techniques, ONE per commit, each byte-exact gated
(both flavors build + `cargo test --release --lib` 0-fail + proptest≥60k + tri-oracle
gzip/flate2/libdeflate/igzip × corpora × T1/T4/T8 sha-identical) THEN paired cyc/byte
A/B vs prior commit AND vs igzip via `_gzippy_vs_igzip_paired_guest.sh`:
1. Loop-level restructure: consume litlen BEFORE the len/EOB classify; preload
   next-litlen + next-dist every iteration (igzip loop_block). HAZARD: collides with
   gz's X2 reclass contract (RECLASS packets must be handed back un-consumed) — needs a
   cheap per-iter (bitbuf,bitsleft) spill+restore; MEASURE the spill cost on the hot
   literal path before committing (it may regress).
2. igzip speculative unconditional store + single discriminator branch (collapse the
   cmp 0x100 / cmp 0xff / test 0x2000000 chain).
3. Update `run_contig_ref` / `run_contig_ref_biased` in LOCKSTEP with every asm change
   (X1-X5 exit-state + IN_MARGIN bit-exact-refill contract).

## TOOLING STATE (for next turn)
- Instrument: `/root/distpreload-harness/_gzippy_vs_igzip_paired_guest.sh` (committed
  in scripts/bench). Re-run gap: `PIN=4 REPS=21 CORPORA="silesia monorepo nasa" /root/distpreload-harness/_gzippy_vs_igzip_paired_guest.sh`
- Symboled native binary for perf/annotate: `/tmp/symtarget/release/gzippy` on guest
  (byte-correct, built from gz-fullrewrite @ 2b10aa48; rebuild after any kernel edit).
  Profile: `perf record -F6000 ... ; perf annotate -s ...run_contig`.
  NOTE: build needs strip=false debug=2 WITHOUT force-frame-pointers (FP collides with
  the register-hungry asm kernel: "inline assembly requires more registers").

## DONE-CRITERION (do NOT self-bless; report as gated-HYPOTHESIS + re-verify cmd)
gzippy T1 cyc/byte ≤ igzip with paired p<0.01 + bootstrap CI excluding 0 + margin
surviving the bandwidth stressor, on silesia AND ≥1 more corpus; + wall-time
confirmation if a quiet window appears. AMD/Zen2 replication owed for LAW.
