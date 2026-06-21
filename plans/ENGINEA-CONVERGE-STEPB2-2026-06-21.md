# ENGINE-A CONVERGENCE — STEP B-2 RESULTS — 2026-06-21

**Branch:** kernel-converge-A. **Scope:** macOS aarch64 (Apple Silicon, quiet).
**Build:** `--no-default-features --features gzippy-native`, path=ParallelSM, -p1.
**Instrument:** `scripts/bench/standing/enginea_converge_mac.py` (Gate-0: byte-exact
all arms + same /dev/null sink + ParallelSM routing; Gate-1: best-of-13, instr/B
load-immune, cyc/B HYPOTHESIS-tier). Frontier comparator = libdeflate-gunzip.
**NOT-YET-LAW:** single-arch; AMD + Intel-engine-A-asm-off owed.

## Increments banked

| # | change | verdict | byte-exact |
|---|---|---|---|
| 1 | `LitLenTable::TABLE_BITS` 12→11 on aarch64 (= libdeflate `LITLEN_TABLEBITS`) | **WIN** (instr) | yes |
| 2 | `DistTable::TABLE_BITS` 9→8 on aarch64 (= libdeflate `OFFSET_TABLEBITS`) | **TIE** (faithful conv.) | yes |

Both arch-conditional (x86 keeps 12/9 — its production path is the run_contig asm; the
x86 asm kernel's `DistTable==9` invariant assert is now x86-gated). Commits:
`ba282489` (inc1), `5b04e1c5` (inc2). Both on origin/kernel-converge-A.

## Before → After (instr/B + cyc/B, best-of-13, /dev/null, gz vs libdeflate)

| corpus | base i/B | HEAD i/B | base i-ratio | **HEAD i-ratio** | base cyc-ratio | HEAD cyc-ratio |
|---|---|---|---|---|---|---|
| silesia | 9.827 | **9.630** | 1.282 | **1.257** | 1.235 | 1.225 |
| monorepo | 6.811 | **6.742** | 1.224 | **1.214** | 1.252 | 1.249 |
| nasa | 3.396 | **3.370** | 1.063 | **1.055** | 1.284 | 1.293 |
| decomp_literal | 20.669 | **20.471** | 1.256 | **1.243** | 1.133 | 1.115 |
| decomp_backref | 1.831 | 1.832 | 1.076 | 1.072 | 1.503 | 1.578 |

- **instr/B (deterministic, load-immune — the gated screen):** drops on every real
  corpus; silesia −2.0% (gap **1.282→1.257**, Δ≫spread 0.13–0.26%), monorepo −1.0%,
  nasa −0.8%, literal extreme −1.0%. The aarch64 T1 frontier **instr gap shrinks**.
- **cyc/B (wall proxy, HYPOTHESIS-tier — quiet box):** silesia c-ratio 1.235→1.225
  (Δ within spread → **wall TIE-trending-better**); others within noise. The wall is a
  TIE; the instruction-count win is real and gated.

## Gates passed
- Byte-exact: sha==`gzip -d` on silesia/monorepo/nasa/decomp_literal/decomp_backref at
  T1/T4/T8 (engine A is shared across T; the resumable careful tail is untouched).
- 943 lib tests pass (incl `three_oracle_silesia`, `resumable_correctness`,
  `trace_parity_*`, routing CRC-stress).
- Routing: GZIPPY_DEBUG=1 → path=ParallelSM.

## What remains (next increments — see ENGINEA-CONVERGE-DESIGN)
- **The top remaining lever** is the per-symbol LITERAL path: engine A's 8-deep
  packed-write literal ladder vs libdeflate's 3-literal individual-write + the
  conditional-vs-unconditional refill cadence (design divergences #2+#3, entangled).
  This is the heroic byte-exact restructure of the fastloop body; it directly attacks
  the +4.21 instr/symbol literal excess but risks regressing literal-heavy data, so it
  needs a dedicated hard-gated pass (decomp_literal + silesia + resumable contract).
- Cross-ISA LAW: Intel engine-A-asm-off + AMD/Zen2 replication owed.
