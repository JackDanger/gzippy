# BEAT-IGZIP-T1 — DURABLE STATE

Mission: make gzippy-native (pure-Rust, FFI-off) **T1 single-member gzip DECODE**
measurably FASTER than igzip (ISA-L), byte-exact, gated. Single-arch Intel = NOT LAW
(AMD/Zen2 replication owed). T1 single-core only — no T4/T8 extrapolation.

Branch: `perf/igzip-full-rewrite`. Mac worktree (edit/commit/push only — aarch64,
CANNOT run the asm): `/Users/jackdanger/www/gzippy/.claude/worktrees/agent-a8069a92d914fcef3`.
Guest (ONLY x86_64/BMI2 measure box): `ssh -J 10.0.0.100 root@10.30.0.199`.
Guest worktrees: gzippy(B)=/root/gz-fullrewrite (kernel 2b10aa48 dist-preload),
baseline=/root/gz-baseline (8383a2eb). igzip=/usr/bin/igzip (ISA-L 2.31.1).
Harness on guest: /root/distpreload-harness/.

## COMMITS THIS MISSION
- 2c135d07 — deliverable #0: commit orphaned paired harness (analyzer+memstress+driver) to scripts/bench.
- 2e01dd4f — deliverable #1: add igzip arm `scripts/bench/_gzippy_vs_igzip_paired_guest.sh`.
- (HEAD)   — Step-B technique #1: fuse `lea-1`+`shl3` shift into one `lea [t3*8-8]`.
            BYTE-EXACT (sha 3 corpora×T1/T4/T8, proptest 60k, c2/c3 asm-vs-ref diffs).
            Δinstr/byte=-0.186 (silesia, deterministic + paired); cyc/byte TIE
            (silesia medΔ=-0.040, CI=[-0.0475,-0.0216] excl 0 but p=0.018 fails p<0.01;
            nasa wash). KEPT on byte-exact license; gap to igzip NOT closed (still +39.5%).

## STEP-B TECHNIQUE LOG (gated, Intel trainer cpu4, N=21 paired, /dev/null sink)
| # | technique | byte-exact | Δinstr/B (sil) | cyc/B medΔ (sil) | p | verdict | kept? |
|---|-----------|-----------|----------------|------------------|---|---------|-------|
| 1 | fuse shift `lea[t3*8-8]` (was lea-1+shl3) | PASS | -0.186 | -0.040 [CI-0] | 0.018 | TIE (fails p<0.01) | YES (byte-exact) |
  Re-verify #1: BIN_A=<prior> BIN_B=<new> PIN=4 REPS=21 CORPORA="silesia nasa"
  SKIP_STRESS=1 bash /root/distpreload-harness/_distpreload_paired_guest.sh
  LESSON (confirms advisor #1): removing a PREDICTABLE, load-shadowed instr drops
  instr/byte but NOT cyc/byte proportionally — the shl was overlapped behind the table
  load. To move cyc/byte the lever must cut the CRITICAL-PATH ALU (the shrx-fed
  classify dep chain / mispredicting branch), not just retired-instruction count.

## GATE-2 CAUSAL PERTURBATION — IS THE UNPACK/CLASSIFY CHAIN ON THE CRITICAL RECURRENCE? (2026-06-18)
DECISIVE TEST that gates whether rewrite B (igzip-style fat DIRECT table, ~0 unpack)
is worth funding. Method (STRONG tier — Gate-2 causal perturbation): in a THROWAWAY
worktree (/root/gz-perturb, base 2b10aa48 — identical shrx→cmp critical segment;
REMOVED after run), inserted K serially-DEPENDENT, value-NEUTRAL ALU ops
(`rol $1,%r10`/`ror $1,%r10` pairs, all on t5) BETWEEN the trailing-symbol extraction
`shrx {t5},{t5},{t4}` and the literal/non-literal discriminator `cmp {t5:e},255`,
extending the loop-carried recurrence that feeds the discriminator branch.
CONTROLS PASSED: all 4 builds byte-exact (sha==zcat silesia ref) + kernel-engaged
(entries=24128 identical) + distinct binaries; disasm-verified the chain is a genuine
DEPENDENT serial chain on %r10 sitting exactly between the shrx and `cmp $0xff,%r10d`
+ `ja` (rol/ror count scales 8→10→12→16 = +K); paired self-test (A2-A1) PASS.
Harness: `_distpreload_paired_guest.sh`, PIN=4 REPS=21 SKIP_STRESS=1, /dev/null sink.

SLOPE TABLE (medΔ cyc/byte vs K=0, all p=6.412e-05, CI excludes 0):
| K (dep ops) | silesia medΔ (per-op)   | nasa medΔ (per-op)     |
|-------------|-------------------------|------------------------|
| 2           | +0.280 (+4.6%) [0.140]  | +0.070 (+2.1%) [0.035] |
| 4           | +0.686 (+11.3%) [0.171] | +0.204 (+6.1%) [0.051] |
| 8           | +1.380 (+22.8%) [0.172] | +0.375 (+11.2%) [0.047]|

VERDICT: **PROPORTIONAL slope on BOTH corpora** (monotone, ~linear; silesia
doubling-K → ~2.0× Δ, ~0.17 cyc/B per op; nasa ~0.05 cyc/B per op). ⇒ the
shrx-fed t5-extraction → discriminator chain (the gz-specific UNPACK/classify the
STATE'd ~21% region) IS on the critical recurrence — NOT load-shadowed (contrast
technique #1's load-shadowed shl, which TIE'd). Eliminating that extraction
(rewrite B: direct table yields byte+len, ~0 unpack) therefore has REAL cyc/byte
headroom. The unpack cost is concentrated on entropy-DENSE/literal-heavy data
(silesia 0.17/op) vs backref-heavy/low-entropy (nasa 0.05/op) — i.e. B's payoff is
LARGEST exactly where gzippy's gap is largest (silesia +39.5%).

CAVEAT (Gate-2 rule: slow-down slope ≠ speed-up ceiling): this proves the chain is
ON the critical path (necessary), NOT how much B's shorter chain recovers. The
removal-ORACLE that BOUNDS the win already exists and is banked: igzip's
decode_huffman kernel = 3.98 cyc/B vs gzippy run_contig 5.32 cyc/B (silesia, Step A
iii) ⇒ up to ~1.34 cyc/B recoverable, and igzip pays ~0 unpack. So: perturbation
(chain on critical path) + igzip removal-oracle (1.34 cyc/B ceiling) ⇒ B is
JUSTIFIED. Single-arch Intel = NOT-YET-LAW (AMD/Zen2 owed). monorepo 3rd-corpus owed
(optional; 2-corpus spread already concordant).

RECOMMENDATION FOR USER R3 (fund rewrite B?): **YES, justified** — gated proportional
slope (both corpora) + a measured igzip removal-oracle ceiling (~1.34 cyc/B). BUT do
the CHEAPER step first (pre-registered NEXT #1: flag-bit discriminator — same critical
chain, ~1/10th the work, no table re-layout): if a single `test entry,FLAG; jnz` off
the LOAD (no shrx-extract) recovers a significant fraction of the slope, B's full
table-format rewrite may be unnecessary. Sequence: flag-bit (NEXT #1) → re-measure →
THEN decide B with the residual.

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

## STEP A — AIM CONFIRMED (2026-06-18, guest trainer cpu4, symboled bin rebuilt today)
Localization re-run fresh; all three sub-checks PASS and converge on "the excess is in
the hot run_contig classify+decode loop, addressable."

(i) RECLASS RATIO — cold exits are NOT the cost. GZIPPY_VERBOSE=1 GZIPPY_ASM_STATS=1:
| corpus   | entries | exit_reclass(tag0) | reclass_dist | reclass_eob | asm_bytes/entry |
|----------|---------|--------------------|--------------|-------------|-----------------|
| silesia  | 24128   | 0                  | 21330        | 2796        | ~8770           |
| monorepo | 8299    | 0                  | 8130         | 166         | ~6118           |
| nasa     | 25399   | 0                  | 25037        | 351         | ~8077           |
  Generic invalid/oversize exit (tag 0) = 0 everywhere. The dominant exit is
  reclass_dist (subtable/raw0 dist → Rust), BUT asm_bytes/entry is ~6-9 KB: the kernel
  decodes thousands of CLEAN bytes between each bail, so per-byte cold-exit overhead is
  negligible. The excess instr/byte is in the hot loop. => loop_block port is aimed right.

(ii) PER-REGION (perf annotate -F4000, silesia, hottest run_contig instrs, self%):
  - Packed-symbol UNPACK ALU chain (gz's multi-sym short-entry format tax, no igzip
    counterpart): test 0x2000000 3.27, mov t2,t1 4.31, shl $3 (shift=8*(cnt-1)) 3.81,
    and 0x1FFFFFF 2.96, mov 2.23, lea-1 1.38, shrx 1.12, cmp 0xff 1.92  => ~21%
  - Two table-load preloads: mov(%r11,%r14,4) 5.63 + mov(%r11,%r12,4) 5.55 => ~11%
    (IRREDUCIBLE — igzip's decode_next_sym has the identical 2 loads)
  - MOVDQU backref copy: load 4.69 + store 4.10 + jle 2.03 => ~11%
  - Non-literal arm: cmp 0x100 (EOB) 5.12, cmp 0x200 1.28, dist preload/decode/copy body

(iii) SYMBOL-RESTRICTED KERNEL-vs-KERNEL cyc/byte (perf stat cycles × self%, silesia):
  - gzippy run_contig self=86.64%, total=1,302,094,139 cyc / 211,968,000 B => 5.32 cyc/B
  - igzip decode_huffman_code_block_stateless_04 self=90.31%, total=934,030,780 cyc
    => 3.98 cyc/B
  - KERNEL GAP = +1.34 cyc/B (+33.7%) — tracks the whole-process silesia +39.5%, so the
    excess really is IN the kernel. Whole-proc instr/byte: gzippy 13.73 vs igzip 11.38
    (+2.35), consistent with banked +~2.3.

  VERDICT (Step A): the +1.34 cyc/B kernel deficit decomposes as ~21% packed-symbol
  UNPACK ALU + ~11% backref copy + ~11% irreducible table loads. The single addressable
  lever with NO igzip counterpart is the packed-multi-symbol UNPACK chain (cnt extract,
  shift=8*(cnt-1), shrx, masks). igzip's table format yields the decoded byte(s)+length
  DIRECTLY, paying ~0 unpack ALU — that ~21% is the instruction excess. NOTE: attacking
  it = changing the short-entry TABLE FORMAT (build side in lut_huffman) in lockstep, a
  STRUCTURAL change, not a peephole; high byte-exact risk, must be one gated commit.

## NEXT (planned, in priority order) — for the iteration phase (deliverable #2)
REVISED after technique #1's TIE: the prize is CRITICAL-PATH cyc/byte, NOT retired
instruction count. Technique #1 cut -0.186 instr/byte yet TIE'd cyc/byte because the
removed shl was load-shadowed (overlapped behind the table-load latency). Peephole
instruction-shaving of already-overlapped ops will keep TIE-ing. The remaining cyc/byte
deficit (+1.34 cyc/B kernel) lives in (a) the data-dependent classify dep-chain that
FEEDS the load and the cmp (the `shrx`-fed trailing-symbol extraction → `cmp 255`
branch; negative ΔIPC says this mispredicts/serializes), and (b) the ~21% packed-symbol
UNPACK ALU that has no igzip counterpart. So:
1. (HIGHEST VoI) Attack the CRITICAL PATH, not the instruction count. The dep chain is
   short_tbl load -> (bc/cnt/shift extraction) -> shrx -> cmp 255 -> branch. Shorten the
   chain LENGTH or break the misprediction: e.g. derive the literal-vs-nonliteral
   discriminator from a flag BIT already in the loaded entry (a single `test`, no
   shrx-extract), so the branch resolves off the load directly instead of waiting on the
   extraction chain. Requires a table-format bit (build side lut_huffman) — STRUCTURAL,
   one gated commit, ref in lockstep. Pre-register success = cyc/byte medΔ<0 p<0.01.
2. The packed-multi-symbol short-entry format itself: evaluate whether multi-symbol
   packing (decode 2-3 lits/iter, pay UNPACK ALU) actually beats igzip's
   one-symbol-per-iter + fatter DIRECT table (decoded byte+len in the entry, ~0 unpack).
   This is the core convergence question — likely the real lever, but a large rewrite +
   STRATEGIC fork: escalate to supervisor/user before committing the rewrite (R3).
3. Update `run_contig_ref` / `run_contig_ref_biased` in LOCKSTEP with every asm change
   (X1-X5 exit-state + IN_MARGIN bit-exact-refill contract).
GATE (each technique, ONE per commit): byte-exact (sha 3 corpora×T1/T4/T8 + proptest≥60k
+ c2/c3 asm-vs-ref diffs) THEN paired N≥21 cyc/byte vs prior AND vs igzip. Build native
flavor: `RUSTFLAGS="-C target-cpu=native" cargo build --release --no-default-features
--features pure-rust-inflate`. NOTE: local Mac pre-commit hook is BROKEN (uninitialized
vendor submodules) — commit on the guest, or `git commit --no-verify` for non-perf
metadata; CI still runs the real checks.

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
