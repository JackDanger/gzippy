# INTEL CROSS-VALIDATION of the engine-A wire-in (kernel-converge-A @834ba516)

Scope: **Intel i7-13700T LXC (neurotic guest, x86_64), NOT-YET-LAW for cyc/wall.**
Deterministic primitive = `cpu_core/instructions/` via `perf stat`, pinned to a
single P-core (`taskset -c 4`), `-p1`, `/dev/null` sink BOTH arms, interleaved
A,B,A2 per rep, N=15. The box is UNFROZEN (`intel_pstate/no_turbo=0`,
governor=`powersave`, LXC read-only) → cyc/B is NOISY and reported as CONTEXT
ONLY; the verdicts ride on the deterministic instruction count (per-arm spread
0.1–1.5%).

Two binaries built fresh on the guest at sha `834ba516`, RUSTFLAGS
`-C target-cpu=native`, both `build-flavor=parallel-sm+pure`, both
`path=ParallelSM`:
- **gz-asmoff** — `pure-rust-inflate` ON, **`asm-kernel` OFF** → flat pure-Rust
  **engine A** is the active clean contig path on x86 (the cfg
  `not(all(feature="asm-kernel", target_arch="x86_64"))` is TRUE).
  INCANTATION: transient one-line edit of the ephemeral guest checkout —
  `pure-rust-inflate = ["rpmalloc-caches", "asm-kernel"]` →
  `pure-rust-inflate = ["rpmalloc-caches"]`, then
  `cargo build --release --no-default-features --features gzippy-native`, then
  `git checkout Cargo.toml`. Keeps `pure-rust-inflate` ON (all ~40 literal
  `feature="pure-rust-inflate"` gates behave exactly as production) and toggles
  ONLY `asm-kernel`. (A standalone additive Cargo feature is NOT viable: the
  source relies pervasively on the literal `feature="pure-rust-inflate"`,
  including `not(...)` arms that would flip.)
- **gz-asmon** — unmodified tree, `--features gzippy-native` (`pure-rust-inflate`
  pulls `asm-kernel`) → the x86 BMI2 `run_contig` asm is the clean path; engine A
  is compiled OUT (this is Intel PRODUCTION today).

## D1 — BYTE-EXACT on x86 (HARD GATE) — PASS

- sha==zcat grid 18/18 OK: {gz-asmoff, gz-asmon} × {silesia, monorepo, nasa} ×
  {T1, T4, T8}.
- Kill-switch byte-exact 3/3 OK: `GZIPPY_FLAT_CLEAN=0` (engine B) on gz-asmoff,
  silesia/monorepo/nasa T1.
- **Gate-0c non-inert proof (engine A PROVABLY ran on x86):** gz-asmoff silesia
  T1 → `flat_contig calls=2807 bytes=211730617` (~99.9% of output via engine A);
  with `GZIPPY_FLAT_CLEAN=0` → `calls=0` (engine B, proves the kill-switch
  routes); gz-asmon → no flat_contig line (engine A cfg'd out, run_contig active).
- Seam suite PASS (15/15) on x86 INCLUDING
  `seam_bounded_fastloop_resumes_byte_exact_at_every_budget` (the n_max
  boundary-fuzz differential) + the `tests::seam_crossing` suite, both asm-OFF
  and asm-ON trees.

## D2 — engine A (flat) vs engine B (two-level), kill-switch, SAME binary — engineA WINS, CROSS-ISA LAW

| corpus | A engineA instr/B | B engineB instr/B | B/A (instr) | Δ vs spread | cyc B/A (noisy) |
|--------|------------------:|------------------:|------------:|------------:|----------------:|
| silesia  | 13.064 | 22.366 | **1.712×** | +71.2% ≫ 0.46% | 1.354× |
| monorepo |  8.766 | 13.916 | **1.587×** | +58.7% ≫ 1.21% | 1.261× |
| nasa     |  4.385 |  7.014 | **1.599×** | +59.9% ≫ 0.64% | 1.303× |

Self-test A2/A = 0.999–1.0004 (licenses the ratios). Engine A is 1.59–1.71×
cheaper in instructions on x86, CI-disjoint on every corpus — **SAME DIRECTION as
the macOS-aarch64 1.80× whole-program silesia** (origin @834ba516, prior gated).

⇒ **CROSS-ISA LAW (Intel asm-off + macOS aarch64): the pure-Rust FLAT engine A
beats the two-level engine B.** Both ISAs, both CI-disjoint. Strongest validation
available without AMD. (AMD/Zen2 still owed for full x86-microarch LAW, but the
pure-Rust finding reaching Intel+macOS IS cross-ISA.)

## D3 — ⭐ engine A (flat pure-Rust, asm-OFF) vs run_contig (x86 BMI2 ASM, asm-ON) — engineA ≥ asm

| corpus | A engineA(asmoff) instr/B | B run_contig(asmon) instr/B | B/A (instr) | Δ vs spread | verdict | cyc B/A (noisy) |
|--------|--------------------------:|----------------------------:|------------:|------------:|---------|----------------:|
| silesia  | 13.054 | 13.617 | **1.043×** | +4.31% > 0.27% | engineA WINS | 1.013× |
| monorepo |  8.766 |  8.903 | 1.016× | +1.57% ≈ 1.47% | TIE/marginal-WIN (at LXC floor) | 1.099× |
| nasa     |  4.384 |  4.496 | **1.026×** | +2.55% > 0.60% | engineA WINS | 1.022× |

Self-test A2/A ≈ 1.0000. **On the deterministic instruction primitive, flat
pure-Rust engine A is NEVER worse than the x86 BMI2 hand-asm run_contig — a clear
WIN on silesia (+4.3%) and nasa (+2.6%), a marginal WIN/TIE at the resolution
floor on monorepo (+1.6%).** Noisy cycles agree directionally (cyc B/A ≥ 1.01
everywhere, i.e. engine A uses ≤ the asm's cycles).

### VERDICT: RETIRE-asm SUPPORTED (instruction-decisive); frozen-box cyc/wall + AMD OWED before the deletion
The x86 BMI2 asm kernel's raison d'être is cycle/IPC, the one axis this unfrozen
LXC cannot resolve to LAW — so this is RETIRE-LEANING, not an unconditional
RETIRE. But the structural finding is robust: engine A ties-or-beats the hand-asm
on the deterministic primitive, AND the same engine-A superiority replicates
across ISAs (D2). If engine A ≥ run_contig holds on a frozen box, the large
hard-to-maintain x86 BMI2 asm can be RETIRED and engine A becomes the SINGLE
portable flat clean kernel for ALL arches (the sole-path end-state).
AMD/Zen2 strengthens the case structurally: BMI2 (`pext/pdep/bzhi`) is MICROCODED
on Zen2, so run_contig should do RELATIVELY WORSE there while engine A is portable
scalar — engine A likely widens its lead on AMD. NOT-YET-LAW: owed = (1) frozen
Intel or AMD cyc/wall A/B, (2) AMD/Zen2 replication.

## Reproduce
```
# guest: build both binaries at the sha
bash scripts/bench/kernel-ab/intel_xval_build.sh         # -> /dev/shm/ixv/{gz-asmoff,gz-asmon}
bash scripts/bench/kernel-ab/intel_xval_d1grid.sh        # D1 byte-exact + non-inert
N=15 bash scripts/bench/kernel-ab/intel_xval_perf.sh     # D2/D3 instr+cyc A/B -> /dev/shm/ixv_perf.csv
python3 scripts/bench/kernel-ab/intel_xval_analyze.py    # verdicts (point it at the pulled csv)
```
