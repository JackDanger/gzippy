# LOGS-T1 TABLE-BUILD — causal gate + close (2026-06-26)

**VERDICT: table-build IS the gated logs-T1 lever (causally confirmed, NOT another
mislead-by-attribution). A byte-exact per-block table-build CACHE closes 23–31% of the
gz-vs-igzip logs gap with NO cross-corpus regression — SHIPPED (size-1 MRU). The fuller
caching ceiling (LRU-8 → ~41% hit / +3.0% logs) is REAL but not free-on-miss
(regresses silesia ~0.3%), so it is recorded as a future adaptive-dispatch lever, not
shipped. The residual gap is the per-BUILD cost itself (gz `rebuild_from` ~2.4× igzip
`make_inflate_huff_code`) — the next, build-convergence, lever.**

## SCOPE STAMP
- cell: **logs-T1** (LOSS) + logsbig (scaled logs); silesia/nasa/monorepo = no-regression controls.
- arch: Intel **i7-13700T raptorlake** (trainer, `ssh -J neurotic root@REDACTED_IP`),
  **core 0**, taskset-pinned. Box CONTENDED (load 2.7–3.5) — the PAIRED interleaved
  ratio + contention-invariant certification are the decisive load-immune arms;
  cyc/B ABSOLUTE spread-gate VOID-QUIET (owed a quiet box for LAW).
- commit: reimplement-isa-l + this branch (`feat/logs-t1-tablebuild` @b86944d8);
  gzippy-native (`--no-default-features --features pure-rust-inflate`,
  `RUSTFLAGS="-C target-cpu=native -C debuginfo=1 -C strip=none"`).
  **final bin sha 6ca6c4cdb5c988748e3cb72e1a170d5878e8bb358b708676ee9d07c735ff9abc**
  (`/dev/shm/s2t/release/gzippy`). **path=ParallelSM threads=1**, build-flavor=parallel-sm+pure.
- corpora `/dev/shm/{logs,logsbig,silesia,nasa,monorepo}.gz`; igzip 2.31.1; oracle `gzip -dc`.
- instrument: `fulcrum abmeasure` (interleaved paired A/B + igzip comparator + sha gate);
  byte-transparent env knobs (all self-validated non-inert below). date: 2026-06-26.
- **AMD + aarch64 replication OWED** (NOT-YET-LAW) — solvency/AMD NOT touched (llama).

---

## STAGE 1(a) — PERTURBATION (Gate-2 critical-path test) → PASS, table-build is ON the critical path

Knob `GZIPPY_TBUILD_MULT=N` (marker_inflate.rs, NIGHT35) runs the per-block litlen LUT
build N times into the idempotent state (N-1 throwaway rebuilds + the real one) — same
output bytes, redundant build passes. **Self-validation (Gate-0):**
- byte-transparent: sha(mult=1)==sha(mult=3)==`gzip -dc` on logs (b5c6632f…).
- NON-INERT: perf-stat instructions on logs mult=1 = **787.5M** → mult=3 = **974.5M**
  (+93.5M per extra litlen build = +11.9% instr/build pass — the extra passes provably ran).

`fulcrum abmeasure`, logs, core 0, N=15, paired interleaved (base = mult=1):

| perturb | after/base (wall) | paired | sign-test | contention-invariant | A/A |
|---|---:|---:|---:|---|---|
| **mult=2** (1 extra build) | **1.0975 (+9.75%)** | 0+/15− | p=6.1e-5 | cross-stratum 0.0028 ≤ 0.25·eff ⇒ **FLAT** | SYMMETRIC |
| **mult=3** (2 extra builds) | **1.1904 (+19.0%)** | 0+/15− | p=6.1e-5 | range 0.0051 ⇒ **FLAT** | SYMMETRIC |

- **MONOTONIC + LINEAR + PROPORTIONAL: ~+9.5% wall per extra full litlen build.** The wall
  responds to a known scaling of table-build time ⇒ **table-build is ON the logs-T1
  critical path** (Gate-2 PASS). This is the test the copy lever FAILED (copy was slack);
  table-build is NOT slack. **Each per-block litlen build ≈ 9.5% of the logs-T1 wall.**
- The cyc/B absolute spread-gate VOIDs (contended box) but the paired direction +
  cross-stratum FLAT certification is load-immune and decisive.

## STAGE 1(b) — REMOVAL-ORACLE via TABLE CACHING → hit-rate + ceiling

A block whose dynamic-Huffman header is byte-identical to a previously-built one can
reuse the already-built LUT (`rebuild_from` is a deterministic fn of (code_lengths,
multisym); decode only READS the table). Cache stats (`GZIPPY_TBUILD_CACHE_STATS=1`,
self-validated non-inert via the hit/miss/distinct counters):

| corpus | litlen builds | size-1 (MRU) hit | unbounded distinct-ceil | LRU-K knee |
|---|---:|---:|---:|---|
| **logs** | 2906 | **21.1%** | 42.9% | K=8 → 41.1% |
| **logsbig** | 14773 | **21.7%** | 51.1% | (K=8 ≈ 41%) |
| silesia | 3364 | 2.3% | — | (low reuse) |
| nasa | 421 | 16.4% | 16.4% (all consecutive) | — |
| monorepo | 184 | 9.2% | — | — |

LRU-K hit-rate sweep on logs (key-only simulator): K1=21.1, K2=28.5, K4=36.4,
**K8=41.1**, K16=41.9, K32=42.0 → **K=8 is the knee** (≈ the 42.9% unbounded ceiling).
So logs blocks reuse headers, but mostly NON-consecutively → a small LRU ~doubles the
size-1 reuse. (Reconciles the "~60% of the gap" hope: table-build is ~9.5%/build × the
hit-able fraction; even full reuse caps near the unbounded ceiling, not 100%.)

---

## STAGE 2 — CLOSE: ship the SIZE-1 (MRU) cache; LRU-8 measured-but-not-shipped

### SHIPPED: size-1 (last-header) cache — FREE on a miss
`LutLitLenCode` keeps the code-length key of the table resident in `self.table`; a block
with the identical key (+ multisym) skips `rebuild_from` with ZERO copy. Miss cost = one
key compare. `GZIPPY_TBUILD_CACHE_OFF=1` disables (the A/B base arm).

- **Byte-exact (BLOCKING) PASS:** sha(cache ON)==sha(cache OFF)==`gzip -dc` on
  logs/logsbig/silesia/nasa/monorepo (5/5), AND **949/949 lib tests pass** (corpus
  multi-oracle differentials, fixed+dynamic Huffman, marker/ring path, CRC stress,
  routing) with the cache ON by default.

**Gated wall (`fulcrum abmeasure`, N=15, core 0, base = `GZIPPY_TBUILD_CACHE_OFF=1`):**

| corpus | after/base | Δ | paired | gap closed (gz/igzip) | verdict |
|---|---:|---:|---:|---:|---|
| **logs** | **0.9751** | **+2.5%** | 15+/0− p=6.1e-5 | **23.1%** | **WALL WIN [CONTENTION-INVARIANT]** (FLAT, A/A SYM) |
| **logsbig** | **0.9735** | **+2.6%** | 15+/0− p=6.1e-5 | **31.0%** | WIN (cyc/B SIG; cert window not quiet) |
| silesia | 0.9989 | +0.1% | 11+/4− NS | — | **TIE (no regression)** |
| nasa | 0.9976 | +0.2% | 10+/5− NS | — | **TIE (no regression)** |
| monorepo | 0.9960 | +0.4% | 9+/6− NS | — | **TIE (no regression)** |

instr/B drops on the win cells (logs 3.566→3.454; logsbig 3.508→3.388 — the skipped
table-build instructions). **Clean: a certified contention-invariant wall win on logs,
+2.6% on logsbig, and TIE (no regression) on every control corpus** — respects the
"caching must be ~free on a miss" gate.

### MEASURED-but-NOT-SHIPPED: LRU-8 (the fuller ceiling)
An 8-slot table-LRU reached the measured 41.1% hit rate on logs and a BIGGER wall win
(**logs +3.0%, logsbig +3.0%, closing 24%/38.5%**, byte-exact, 5/5 sha). **BUT it is NOT
free on a miss** — every miss pays a ~590-cyc table store-copy + an 8-slot find scan, so
on low-reuse silesia (2.3%) it REGRESSED: after/base **1.0034 (−0.3%), paired 0+/15−,
cyc/B Δ −0.0231 SIG**. The store-copy is fundamental (an LRU>1 cannot keep its tables in
`self.table`, which the asm reads at a fixed base). Per the explicit no-regression
mandate, **size-1 ships; LRU-8 is recorded as a future logs-corpus / adaptive-dispatch
lever** (its key-only simulator is retained, stats-gated, for re-verification).

---

## STAGE 3 — VERDICT + tier

- **table-build-is-the-gated-lever-closing-X%:** YES. Perturbation (Gate-2) confirms
  table-build is on the logs-T1 critical path at ~9.5%/build. The shipped size-1 cache
  closes **23.1% (logs, contention-invariant certified) / 31.0% (logsbig)** of the
  gz-vs-igzip gap, byte-exact, no cross-corpus regression.
- **located AND partially cheaply-closable:** the REUSE portion is closed by the cache.
  The RESIDUAL (the per-build cost: gz `rebuild_from` ~2.4× igzip
  `make_inflate_huff_code`/`read_header`, ~60% of the original gap on distinct headers)
  is the NEXT lever — make each BUILD faster (converge to igzip's construction), not
  cache it. The LRU-8 ceiling (+3% logs) bounds how much MORE caching alone can buy
  (≈ +0.5–1.3% over size-1) at the cost of silesia, so build-convergence — not a bigger
  cache — is where the remaining gap lives.
- Tier: **STRONG / load-immune (this arch/commit):** Gate-2 perturbation slope (linear,
  certified FLAT); cache wall win (paired 15/15, logs contention-invariant certified);
  byte-exact (5 corpora × modes + 949 tests). **OWED for LAW:** AMD + aarch64
  replication; quiet-box cyc/B absolute.

## RE-VERIFY (trainer Intel; STRICTLY SEQUENTIAL; /dev/shm; bin 6ca6c4cd)
- perturbation: `fulcrum abmeasure --base-bin $B --after-bin $B --after-env
  "GZIPPY_TBUILD_MULT=2" --common-env "GZIPPY_FORCE_PARALLEL_SM=1" --gz-args "-d -c -p1"
  --rg-cmd "igzip -d -c" --oracle-cmd "gzip -dc" --corpus /dev/shm/logs.gz --n 15 --core 0`
- cache wall: same with `--base-env "GZIPPY_TBUILD_CACHE_OFF=1" --after-env ""` over all 5 corpora.
- hit rate / ceiling: `GZIPPY_TBUILD_CACHE_STATS=1 [GZIPPY_TBUILD_LRU_K=8] $B -d -c -p1 corpus`.
