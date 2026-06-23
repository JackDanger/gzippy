# Cross-Arch LOCATE: x86-T1 run_contig kernel IPC/mispredict on AMD Zen2 (solvency)

**2026-06-22. AMD EPYC 7282 (Zen 2), bare metal solvency (root@REDACTED_IP). Mirrors the
Intel locate (project_kernel_ipc_locate_2026_06_22) to test single-arch over-generalization
of the heroic-rewrite targets. Box checkout 981575f4 (asm_kernel.rs / bmi2.rs BYTE-IDENTICAL
to gh/kernel-converge-A where the Intel c503f171 lived); built /dev/shm/ipc-amd-target,
`--no-default-features --features gzippy-native`, RUSTFLAGS=-C target-cpu=native.**

Comparator: igzip 2.31.0 (ISA-L). Method: taskset -c 8 (away from llama@core23), /dev/null
sink both arms, interleaved N=11, GHz 3.19 stable (boost on, ondemand; cyc/byte freq-invariant
anyway). llama NEVER paused (freq-pinned cyc/byte is load-robust). No host mutations.

Raw data: box /root/gzippy/plans/kernel-ipc-locate-amd-data/{amd_ipc.csv,misp.data,misp_m.data};
mac /tmp/amd_ipc.csv. Harness /dev/shm/{amd_ipc_run.sh,amd_ipc_analyze.py}, /tmp/annot.py.

## GATE-0 (PASS)
- sha: gz == zcat == igzip on silesia (028bd002…) AND monorepo. PASS.
- /dev/null both arms; igzip is the same comparator binary.
- non-inert kill-switch (mission step 1): GZIPPY_ASM_KERNEL=0 → engine-B. asm ON 1.215B cyc
  vs OFF 1.70B cyc silesia (~40% diff, byte-exact) ⇒ asm IS the production path AND **run_contig
  WINS on Zen2** despite microcoded PEXT/PDEP. (engine-B runs 4.77B instr vs asm 2.86B — the
  asm does far less work; the microcoded-PEXT penalty does NOT cost it the win.)
- A/A self-test (second gz arm): cyc/B diff 0.04% (mono) / 0.17% (silesia) ≪ gap. PASS.
- spread < 2.5% ≪ gap; GHz 3.188–3.190 all arms. PASS. path=ParallelSM thin-T1 marker_chunks=0.

## RESULT — Zen2 vs igzip (N=11, median)
| corpus  | gz cyc/B | ig cyc/B | gap   | gz IPC | ig IPC | gz instr/B | ig instr/B | gz misp/KB | ig misp/KB | gz br/KB | ig br/KB |
|---------|----------|----------|-------|--------|--------|-----------|-----------|-----------|-----------|---------|---------|
| silesia | 5.764    | 5.091    |+13.2% | 2.340  | 2.242  | 13.49     | 11.41 (+18.2%) | 88.04 | 64.87 (+35.7%) | 2345.8 | 1732.2 (+35.4%) |
| monorepo| 4.397    | 3.481    |+26.3% | 1.971  | 1.962  | 8.67      | 6.83 (+26.9%)  | 81.08 | 54.52 (+48.7%) | 1597.8 | 1031.8 (+54.9%) |

(monorepo SUBSTITUTES for nasa — nasa.gz absent on solvency. monorepo = source tarball,
match-heavy, a designated primary corpus.)

### Intel (for comparison, project_kernel_ipc_locate_2026_06_22)
| corpus  | gap   | gz IPC | ig IPC |
|---------|-------|--------|--------|
| silesia | +20.7%| 2.545  | 2.609 (gz LOWER) |
| nasa    | +30.9%| 2.156  | 2.335 (gz LOWER) |

## THE CROSS-ARCH DIVERGENCE (the headline)
**On Intel, gz IPC < igzip IPC** (mispredict stalls drag gz down → topdown attributed
~50-55% of the gap to BAD-SPECULATION). **On Zen2, gz IPC ≥ igzip IPC** (silesia 2.34 vs
2.24; mono 1.97 vs 1.96). The Zen2 cyc/byte gap is therefore ~ENTIRELY the EXTRA
INSTRUCTIONS/BYTE (RETIRING): instr/B gap (+18.2% sil / +26.9% mono) ≈ cyc/B gap
(+13.2% / +26.3%), with gz's higher IPC even partly ABSORBING the surplus. AMD has no
Intel-TMA topdown buckets (`perf stat --topdown` unsupported on Zen2), so bad-spec % is not
directly bucketable, but the IPC sign-flip is decisive: **Zen2's branch predictor + recovery
absorbs the extra mispredicts that gate the wall on Raptor Lake.** Same +35-55% more
branches/byte and +36-49% more mispredicts/byte as Intel — Zen2 just doesn't pay for them in
stall cycles.

## PER-BRANCH MISPREDICT ATTRIBUTION (HYPOTHESIS-tier — sampled ex_ret_brn_misp, NON-precise)
AMD `ex_ret_brn_misp:pp` unsupported; used non-precise sampling (forward skid ~few instr) +
IBS present but not needed. ASLR off (setarch -R, base 0x555555554000) → file offsets. ALL
samples land in the run_contig loop 0xb5b80–0xb5e80. The SAME branches as Intel are present
and mispredicting (silesia / monorepo top sites):
1. **litlen/EOB `cmp $0x100,%ecx; jb` (0xb5c3b/0xb5c41)** — 10.0% sil / 7.6% mono. (Intel #1.)
2. **conditional 48-bit refill `cmp $0x30,%rsi; jae` (0xb5dd6)** — present in the match path
   (skid shadow b5de0/b5dfe). (Intel #2.)
3. **copy-tail `sub $0x10,%r15; jle` (0xb5e35/0xb5e39)** — 4.0% sil, plus copy loop. (Intel #3.)
4. **copy-length dispatch `cmp $0xf0,%r15; ja` (0xb5e0a/0xb5e11)** — the SINGLE HOTTEST on
   BOTH corpora (11.0% sil / 9.0% mono). On Intel the copy family was nasa-DOMINANT and
   silesia-LIGHT (~5%); on Zen2 the copy/match path mispredicts heavily even on literal-heavy
   silesia. (Micro-divergence in weighting; skid makes exact shares coarse.)

## VERDICT — targets UNIVERSAL, mechanism MICROARCH-SPECIFIC
- **run_contig is the correct T1 kernel on Zen2** (wins vs engine-B ~40%; microcoded PEXT does
  NOT flip it). The heroic rewrite is NOT wrong-kernel-for-Zen2.
- **The rewrite TARGETS transfer** (same branches mispredict, same extra-instruction +
  extra-branch root cause on both arches) — NOT single-arch artifacts.
- **The PAYOFF MECHANISM differs.** Intel gap = bad-speculation (mispredict penalty); Zen2 gap
  = retiring (instruction count). Consequence for fix-class selection:
  - A fix that removes BRANCHES/INSTRUCTIONS (e.g. unconditional/branchless refill — deletes
    `cmp $0x30; jae`; branchless symbol dispatch; branchless copy tail) wins on BOTH arches
    (fewer instr on Zen2 AND fewer mispredicts on Intel).
  - A fix that only improves PREDICTABILITY without cutting instruction count would help Intel
    but be ~NEUTRAL on Zen2.
  - **RANK-2 "cut instr/byte" is the PRIMARY Zen2 lever** (it is essentially the whole Zen2
    gap), where it was only secondary on Intel. RANK-1 "kill mispredict cost" pays on Zen2
    only through the instructions it removes.
- **gate the STEP-2 fix on Zen2 cyc/byte AND instr/byte, not just Intel mispredict counts** —
  else a mispredict-only win could regress/stall on AMD.

## GATES / SCOPE
Gate-0 PASS. Gate-1 PASS (N=11 interleaved, GHz stable, Δ≫spread). Gate-2 N/A (LOCATE; the
STEP-2 fix's optgate is the verdict). Gate-3: this IS the cross-arch arm — Intel + AMD/Zen2 now
both measured; aarch64 (engine-A Rust fastloop, different kernel) still owed. Gate-4 PASS
(gzippy-native, target-cpu=native, sha-verified, path=ParallelSM). Gate-5: macro IPC/retiring
split = STRONG-for-locate (IPC sign-flip is robust); per-branch shares = HYPOTHESIS-tier
(non-precise AMD sampling, forward skid). nasa→monorepo substitution noted.
Box restored: gov=ondemand, boost=1 (never changed), llama Rl (never paused), no leftover pins.
