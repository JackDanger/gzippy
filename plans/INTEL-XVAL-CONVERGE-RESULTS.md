# INTEL CROSS-VALIDATION of the SOLE-PATH clean convergence (kernel-converge-A @cffa61ee)

Validates THIS turn's convergence (clean path = 100% engine A: flat bounded fastloop +
the NEW `decode_clean_careful_flat` resumable tail; engine-B two-level careful loop + its
per-block `lut_litlen` build RETIRED/lazy on the clean path). Previously validated only on
macOS-aarch64; this is the Intel x86 replication arm.

**Box:** Intel i7-13700T LXC (`ssh -J REDACTED_IP root@REDACTED_IP`), x86_64, UNFROZEN
(`intel_pstate/no_turbo=0`, governor=`powersave`, LXC read-only). **Deterministic primitive
= `cpu_core/instructions/`** (per-arm spread 0.13–1.24%); cyc/B is NOISY → reported as
CONTEXT ONLY. `taskset -c 4`, `-p1`, `/dev/null` sink BOTH arms, interleaved A,B,A2 per rep,
N=15. All three binaries built fresh on the guest, `RUSTFLAGS=-C target-cpu=native`,
`build-flavor=parallel-sm+pure`, `path=ParallelSM`:
- **gz-asmoff** — HEAD @cffa61ee, `pure-rust-inflate` ON / `asm-kernel` OFF → CONVERGED engine A on x86.
- **gz-asmon**  — HEAD @cffa61ee, unmodified → x86 BMI2 `run_contig` (engine A cfg'd OUT; Intel PRODUCTION today).
- **gz-hybrid** — PARENT @c4c3cc97 (pre-convergence), `asm-kernel` OFF → engine-A bulk + engine-B's
  two-level careful TAIL + its per-block `lut_litlen` double-build.
asm-off incantation = transient one-line Cargo edit dropping `asm-kernel` from the
`pure-rust-inflate` feature list (per the prior xval), restored after each build.

## D1 — BYTE-EXACT on x86 (asm-off engine A) + non-inert — PASS (D1_FAIL=0)
- sha==zcat grid **18/18 OK**: {gz-asmoff, gz-asmon} × {silesia, monorepo, nasa} × {T1, T4, T8}.
- Kill-switch byte-exact **3/3 OK**: `GZIPPY_FLAT_CLEAN=0` (engine B) on gz-asmoff, silesia/monorepo/nasa T1.
- Seam suite **16/16 PASS** (`cargo test --lib seam`), incl. the NEW
  `seam_flat_careful_tail_byte_exact_and_resumable` + the boundary-fuzz differential.
- Full lib suite: 945 pass / 1 fail / 12 ignored — the lone failure is `bench_cf_logs`
  (a parallel `bench_*` test that read a 0-byte `benchmark_data/logs.txt.gz` while another
  test was regenerating it; PASSES in isolation `--test-threads=1` → fixture-generation race,
  NOT a decode regression, unrelated to the convergence).
- **NON-INERT proof (engine A PROVABLY ran the converged tail on x86):** gz-asmoff silesia T1
  → `flat_contig calls=2807 careful_calls=11 clean_lut_builds=0`. So the NEW
  `decode_clean_careful_flat` flat tail RAN (careful_calls>0) AND the engine-B clean
  double-build is GONE (clean_lut_builds=0). Kill-switch `GZIPPY_FLAT_CLEAN=0` flips it to
  `calls=0 careful_calls=0` (engine A routed OFF). gz-asmon → no flat_contig line (cfg'd out).
  (`clean_lut_builds` is 0 in BOTH engines on a clean stream — it counts only the deep
  careful-fallback build; the kill-switch discriminator is `flat_contig`/`careful_calls`.)

## D2 — the −2% DOUBLE-BUILD DROP replicates on x86 (deterministic instr, Δ≫spread)
converged (@cffa61ee) vs hybrid (@c4c3cc97), BOTH asm-off + `GZIPPY_FLAT_CLEAN=1`:

| corpus | A converged instr/B | B hybrid instr/B | B/A (instr) | Δ vs spread | cyc B/A (noisy) |
|--------|--------------------:|-----------------:|------------:|------------:|----------------:|
| silesia  | 12.577 | 13.047 | **1.0374×** | +3.74% ≫ 0.47% | 1.029× |
| monorepo |  8.474 |  8.753 | **1.0329×** | +3.29% ≫ 1.24% | 1.026× |
| nasa     |  4.267 |  4.388 | **1.0284×** | +2.84% ≫ 0.63% | 1.042× |

Self-test A2/A = 0.99992–1.00005. **The double-build removal saves 2.84–3.74% of whole-program
instructions on x86, CI-disjoint on every corpus — REPLICATES the aarch64 −2.07%/−2.25%, and
is somewhat LARGER on x86.** Gated instruction win (cyc directionally agrees).

## D3 — engine A vs engine B (kill-switch) at HEAD — unchanged-or-better cross-ISA LAW
engine A (flat) vs engine B (two-level), same gz-asmoff binary via `GZIPPY_FLAT_CLEAN`:

| corpus | A engineA instr/B | B engineB instr/B | B/A (instr) | Δ vs spread | prior @834ba516 |
|--------|------------------:|------------------:|------------:|------------:|----------------:|
| silesia  | 12.578 | 22.827 | **1.815×** | +81.5% ≫ 0.35% | 1.712× |
| monorepo |  8.473 | 14.215 | **1.678×** | +67.8% ≫ 1.20% | 1.587× |
| nasa     |  4.268 |  7.189 | **1.684×** | +68.4% ≫ 0.56% | 1.599× |

Engine A's lead over engine B is UNCHANGED-or-SLIGHTLY-BETTER vs the pre-convergence xval
(the A arm got cheaper from the double-build drop). **Cross-ISA LAW (Intel asm-off + macOS
aarch64 1.80×) HOLDS for the converged tail.**

## (bonus) engine A (asm-off) vs run_contig (x86 BMI2 asm, asm-on) — RETIRE-asm-leaning STRENGTHENED
| corpus | A engineA instr/B | B run_contig instr/B | B/A (instr) | Δ vs spread | prior @834ba516 |
|--------|------------------:|---------------------:|------------:|------------:|----------------:|
| silesia  | 12.577 | 13.608 | **1.082×** | +8.20% > 0.38% | 1.043× |
| monorepo |  8.478 |  8.882 | **1.048×** | +4.77% > 1.23% | 1.016× |
| nasa     |  4.267 |  4.498 | **1.054×** | +5.43% > 0.65% | 1.026× |

Engine A's instruction lead over the hand-asm WIDENED (double-build drop applied to the A arm).
Still RETIRE-LEANING not unconditional: the asm's value is cyc/IPC, the one axis this unfrozen
LXC can't resolve. cyc directionally agrees (B/A ≥ 1.04 everywhere).

## STATUS
The converged sole-path clean tail is byte-exact on Intel x86 (asm-off), non-inert (the NEW
flat resumable tail provably runs, the engine-B clean double-build provably gone), the −2%
double-build drop REPLICATES (−2.84%/−3.74%, gated), and the engine-A wins (vs engine B, vs
hand-asm) are unchanged-or-better. Cross-ISA (Intel x86 + macOS aarch64) on the deterministic
instruction primitive. **OWED for full LAW:** AMD/Zen2 (BMI2 microcode) + a frozen/bare-metal
box for cyc/wall (the retire-asm final call). NOT-YET-LAW for cyc/wall.

## Reproduce (guest)
```
bash /dev/shm/intel_xval_converge_build.sh         # -> /dev/shm/ixv/{gz-asmoff,gz-asmon,gz-hybrid}
bash /dev/shm/intel_xval_converge_d1.sh            # D1 byte-exact + non-inert
N=15 bash /dev/shm/intel_xval_converge_perf.sh     # DB/D2/D3 -> /dev/shm/ixv_conv_perf.csv
python3 scripts/bench/kernel-ab/intel_xval_converge_analyze.py /dev/shm/ixv_conv_perf.csv
```
