# AMD T≥2 vs rapidgzip — DEEP LOCATE: RESULTS

**Date:** 2026-06-22. **Branch:** `amd-t2t4-locate` (off `origin/kernel-converge-A` @ `671c5752`).
**Box:** solvency AMD EPYC 7282 Zen2, `root@REDACTED_IP`. **Subject sha:** `671c5752` (built on
box; code == kernel-converge-A HEAD). **Falsifier:** `plans/AMD-T2T4-LOCATE-FALSIFIER.md`
(pre-registered + pushed before any number). Graded against CLAUDE.md gates.

## Gate-0 / Gate-4 (all PASS)
- gz built ON box: `--no-default-features --features gzippy-native`, `-C target-cpu=native`,
  build-flavor=`parallel-sm+pure`, path=`ParallelSM`. gz sha == rg sha == zcat on silesia
  (`028bd002…`) and gz==zcat on monorepo (`0dd50d07…`).
- rg comparator: native ELF `0.16.0` at `/root/gz-base/vendor/rapidgzip/…/rapidgzip`.
  A/A (rg-vs-rg best-of-N): silesia-T4 0.8%, monorepo-T2 1.3% — ≪ the 6.7–8% Δ ⇒ ratios trusted.
- Box FROZEN gov=performance/boost=0 (NOT no_turbo) at 2.8 GHz base, watchdog armed; RESTORED
  at exit (gov=ondemand, boost=1 ALL cores, watchdogs killed, 0 performance govs). Measurement
  cores `8,10,12,14` taskset-pinned away from the user's roaming `llama-completion` (untouched);
  clean A/A confirms no neighbor perturbation on the banked cells.

## Both shapes REPRODUCE at HEAD (frozen, N≥9 interleaved, /dev/null both arms, best-of-N)
| cell | gz/rg wall | gz/rg cyc | shape |
|---|---|---|---|
| silesia-T4 (mask 8,10,12,14) | **1.079** | **1.079** (gz 2.499e9 / rg 2.315e9) | cyc≈wall → **WORK-bound** |
| monorepo-T2 (mask 8,10) | **1.066–1.084** | **1.023** (gz 642e6 / rg 627e6) | wall≫cyc → **SERIALIZATION** |

---

## SHAPE A — AMD-T4 WORK-bound inner-kernel cyc-excess → **GO** (inner-Huffman kernel, open territory)

**Symbol breakdown (perf record cycles, 76k samples gz / 70k rg, silesia-T4):**
- **gz decode kernel = 61.3%**: `marker_inflate::Block::read_internal_compressed` **35.3%**
  (Rust marker/clean fast-loop) + `asm_kernel::run_contig` **26.0%** (asm clean kernel).
  marker resolution `resolve_chunk_markers_on_chunk` 5.6%; table-build 1.7%; memmove 1.6%;
  page-zero 2.85%; block-finder 1.0%.
- **rg decode ≈ 61%**: `deflate::Block<false>::read` 39.2% + anon hot loops (@37/@42.end) 16.9%
  + isal `loop_block`/`decode_len_dist` 4.8%. **`applyWindow` 7.2%** (rg) **> gz marker-resolve
  5.6%** ⇒ gz's marker resolution is CHEAPER than rg's (consistent with the banked apply_window
  win). CRC 2.4%. So the ~8% excess is **WITHIN gz's decode kernel**, NOT markers/pipeline/CRC.

**Per-byte structure (GZIPPY_VERBOSE + ASM_STATS, silesia-T4):** clean asm kernel decodes 137 MB
(65% of output) for 26% cyc ⇒ **4.7 cyc/B**; the window-absent marker loop decodes 74.9 MB (35%)
for 35% cyc ⇒ **11.7 cyc/B = 2.5× the clean kernel**, the single largest+least-efficient bucket.
Marker decode splits (mfast_prof): **mfast 59.8%** (63.8 cyc/ev) / **careful-tail 40.2%** (96.4 cyc/ev).

**Gate-2 — NIGHT35 injector (NON-INERT: `entries=1773`, asm_bytes=137 MB):** injecting N dummy
insns/iter into the clean asm kernel raises the silesia-T4 wall **monotonically + proportionally**:
INJECT 0→1→2→4 (= +4/+8/+16 insn/iter) = **+4.2% / +9.0% / +20.0%** (best-of-N=9, first-rep warmup
discarded). ⇒ the clean asm kernel is **ON the T4 critical path** (~1.25% wall / insn/iter);
speeding the kernel pays. (Same knob at monorepo-T2 = only +5.7% — see Shape B.)

**Annotate (where the kernel cycles concentrate):**
- clean `run_contig`: top 3 hot insns ≈ **36%** of its cycles are per-symbol **classification
  branches/compares** (`cmp $0x100` literal/length boundary, `cmp $0xf0`, refill mask `mov $0x3f`)
  → branch/compare-bound on Zen2.
- marker `read_internal_compressed`: a **non-inlined `call HuffmanCodingShortBitsCached::decode`**
  (6.2% — a concrete codegen divergence; rg inlines its decode) + LARGE_FLAG `test $0x2000000`
  (9.5% across two sites) + marker-tag `or $0x38,%al`.

**VERDICT vs falsifier — GO.** The cyc-excess is the inner window-absent DECODE KERNEL — dominant
(61%, Gate-2-proven on the critical path), NOT distributed across subsystems, NOT marker-resolution
(gz cheaper than rg), NOT pipeline/CRC. Not a Zen2-microcode artifact (PEXT/PDEP ruled out prior).
This is the inner-Huffman kernel front (CLAUDE.md OPEN TERRITORY), same front as the T1-native deficit.
**Tier:** kernel-on-critical-path = STRONG (Gate-2 injector). The WITHIN-kernel sub-bucket ranking
(marker fast loop > clean kernel by cyc/B; the non-inlined huffman-decode CALL; clean-kernel branch
density) = HYPOTHESIS (perf-annotate attribution, no per-sub-region perturbation).
**Top-ranked sub-targets:** (1) inline the marker-loop huffman decode (kill the per-symbol CALL);
(2) shrink the careful-tail (96.4 cyc/ev, 40% of marker decode); (3) the clean-kernel per-symbol
classification branches (Zen2 branch-prediction sensitivity).

---

## SHAPE B — AMD-T2 serialization → **partial NO-GO** (decode at rg-parity; residual = un-localized serial wrapper)

**Decompose (conservation-gated timeline reducer, GATE-0 PASS, gap 0.6%, monorepo-T2, 50.9 MB out):**
- only **5 chunks** (adaptive-shrunk from 4 MiB target; 9.8 MB compressed). Workers **~92% busy**
  (effcores_CPU 1.835/2): tid2 busy 114.9 ms / 3 chunks (95%); tid3 busy 108.0 ms / 2 chunks
  (89%), **idle 13.2 ms cross-chunk** (`pool.pick.wait`); consumer tid1 waits 120 ms (expected,
  in-order). Conservation: busy 224.4 + wait 139.7 + gap 0.7 ms.
- **gz parallel decode is at rg-PARITY:** gz Real-Decode 117.5 ms ≈ rg 120.4 ms (gz slightly
  FASTER). rg's own Theoretical-Optimal 0.1034 vs actual 0.1204 ⇒ rg is also ~86% efficient at
  this 5-chunk granularity. The 5-chunk/2-worker granularity tail is **shared with rg** (both 5 chunks).
- The gz-specific excess is the **SERIAL WRAPPER outside the parallel drive**: gz wall 135.4 −
  decode 117.5 = ~18 ms serial; rg wall 124.9 − decode 120.4 = ~4.5 ms ⇒ **~13 ms gz-specific
  serial overhead** (the wall gap is ~10 ms). This sits BELOW the trace horizon (the GZIPPY_TIMELINE
  drive span is 122 ms; the wrapper is process setup + block-finder bootstrap + finalize/CRC,
  before/after the drive).

**Gate-2 perturbations (both DISCRIMINATING — ruled out, not just attributed):**
- **Decode injector contrast:** monorepo-T2 wall rises only **+5.7%** at INJECT=4 vs silesia-T4's
  **+20%** for the same injection ⇒ T2 has decode SLACK → **NOT decode-work-bound** (≠ Shape A).
- **Resident-output-pool removal-oracle** (`GZIPPY_RESIDENT_OUTPUT_POOL=1`, byte-identical sha):
  **FLAT** — gz 135.4 ms vs gz+resident 135.7 ms (TIE; rg 124.9) ⇒ output-buffer alloc / page-
  zeroing is **NOT the serializer**.
- Drive-trace waits: largest = `wait.block_fetcher_get` 26.5 ms (consumer blocks for the FIRST
  chunk), `apply_window` 20.8 ms, `pool.pick.wait` 19.6 ms.

**VERDICT vs falsifier — partial NO-GO.** No single Gate-2-confirmed fat (>40–50%) gz-vs-rg
serializer this cycle: the parallel decode is at rg-parity, the granularity tail is shared with rg,
and the two suspected regions perturbation-tested (decode-work, alloc/page-zero) are RULED OUT.
The located residual = a small (~13 ms) gz-specific **serial wrapper** outside the instrumented
drive, NOT yet localized to one region. **NEXT (no-phases legitimate next action): BUILD a
phase-timing instrument** that splits the serial wrapper into {input mmap/open, block-finder
bootstrap before first dispatch, finalize/CRC/flush after last chunk}, self-validate it (Gate-0
conservation: phases + drive == wall), then Gate-2-perturb each phase. Faithful-rg convergence
target if the bootstrap dominates: rg's `GzipChunkFetcher` startup (the consumer's 26.5 ms first-
chunk wait suggests gz's block-finder/first-chunk latency may exceed rg's).

---

## Recommended next levers
- **Shape A (GO, primary):** inner-Huffman kernel codegen (OPEN TERRITORY). Rank-1: inline the
  marker-loop huffman decode CALL; rank-2: shrink the careful-tail; rank-3: clean-kernel per-symbol
  branch density on Zen2. Prize ≡ measured Δ (the injector slope sets the sensitivity: ~1.25% wall
  per insn/iter removed from the clean kernel; the marker loop at 11.7 cyc/B is the bigger prize).
- **Shape B (build-first):** phase-timing instrument for the serial wrapper, THEN Gate-2 per phase.
  Decode-work and alloc are Gate-2-ruled-out.

## Rig artifacts (on box, banked-by-reference)
`/root/amd_freeze.sh` (idempotent freeze + bounded auto-restore watchdog), `/root/amd_confirm.sh`
(interleaved best-of-N cyc+wall + A/A), `/root/amd_inject.sh` (NIGHT35 injector slope), `/root/amd_residab.sh`
(resident-pool removal-oracle), `/root/tail_metric.py` (conservation-gated busy/idle reducer, copied
from scripts/parallel_sm_tail_metric.py). gz-head built at `671c5752`; debuginfo variant at
`/root/gz-head-dbg/release/gzippy` (CARGO_PROFILE_RELEASE_STRIP=false DEBUG=2) for symbol/annotate.
