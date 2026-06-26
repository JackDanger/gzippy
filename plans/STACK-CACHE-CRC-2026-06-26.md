# STACK: table-build CACHE + VPCLMULQDQ CRC — dominance scoreboard vs igzip & zlib-ng

Date: 2026-06-26 · Branch: `stack-cache-crc` (combined) · Scope: **Intel i7-13700T
(13th Gen raptorlake), x86_64 ONLY — AMD + aarch64 OWED.**

## What was built

`git worktree` off `reimplement-isa-l`; cherry-picked **both** won levers:
- table-build CACHE (PR #134 / `feat/logs-t1-tablebuild`, 7 commits, `lut_huffman.rs`
  + `marker_inflate.rs` + `chunk_fetcher.rs`) — claimed logs +2.5%.
- VPCLMULQDQ CRC fold (`feat/crc-vpclmul` @424255f2, `crc32.rs`) — claimed logs +6.0%.

Clean cherry-picks (disjoint files). Combined build (`stack-cache-crc` @9f83198):
`RUSTFLAGS=-C target-cpu=native`, default features (`pure-rust-inflate` =
gzippy-native, no C-FFI in decode graph). **Gate-4: `path=ParallelSM`, sha ==
`gzip -dc` on all 5 corpora with BOTH levers active.** Both levers confirmed present
in the binary (cache_key in `lut_huffman.rs`, `crc32_vpclmul` in `crc32.rs`).

---

## STAGE 1 — CRC-KERNEL HARDENING (merge-blocker) — **PASS**

Added `crc_kernel_hardening_lengths_x_alignments` + `prop_crc_fold_matches_both_oracles`
+ `crc32_ref_self_consistency` to `crc32.rs`. The dispatched `crc32_fold` (which
routes to the VPCLMULQDQ kernel on x86_64) is checked against **two** oracles:
`crc32fast` (the trusted crate kernel) AND an **independent bit-serial reflected
CRC32** reference (removes the "both wrap the same crate" blind spot).

Sweep: lengths {0..600 exhaustive, every 128 B/iter fold-by-4 boundary 128k±1 up to
~70 272, the 16 B scalar-tail + <128 fallback edges, aarch64 3-way 3072/3·8 edges}
= **2 232 lengths × 64 alignments (0..63) × 6 seeds**, plus a chained-fold invariant,
plus a **4 000-case proptest**.

**Run on Intel (VPCLMULQDQ active):**
```
CRC-KERNEL-HARDENING: pass=857089 fail=0 lengths=2232 alignments=64 seeds=6 chained_ok=true first_fail=None
test result: ok. 28 passed; 0 failed   (incl. the 4000-case proptest)
```
**857 089 structured cases + 4 000 random cases, ZERO mismatches.** The VPCLMULQDQ
fold is byte-identical to crc32fast AND to the independent reference across every
length×alignment that hits the fold-by-4 body, the 16 B tail, and the <128 fallback.
**Merge-blocker CLEARED.** (Also passes locally on aarch64, exercising the `fold3`
3-way `crc32x` kernel — same 857 089/0.)

---

## STAGE 2 — STACKED WALL (fulcrum abmeasure, N=15, core 0, /dev/null, sha)

Interleaved paired A/B; comparators self-tested (`igzip -d -c` and a zlib-ng
`minigzip -d` wrapper both sha-MATCH `gzip -dc`). zlib-ng built from the vendored
submodule (`vendor/zlib-ng`, ZLIB_COMPAT, Release).

⚠️ **VOID-QUIET caveat (honest):** the trainer was not perfectly idle (median
run-queue 2–3). gz/igzip and gz/zlib-ng are **interleaved-paired within each run** so
contention is common-mode, and the logs ratio replicated across the two independent
runs (1.025–1.029) — but per Gate-1 these are NOT-YET-LAW wall verdicts; treat the
ratios as strong-but-uncertified.

### gz (cache+CRC stacked) vs igzip — wall ratio (after/igzip)
| corpus   | gz wall | igzip wall | **gz/igzip** | verdict |
|----------|--------:|-----------:|-------------:|---------|
| **logs** | 167 ms  | 162 ms     | **1.029**    | **LOSS** |
| silesia  | 671 ms  | 682 ms     | 0.984        | WIN |
| nasa     | 219 ms  | 244 ms     | 0.896        | WIN |
| monorepo | 110 ms  | 116 ms     | 0.952        | WIN |
| logsbig  | 789 ms  | 793 ms     | 0.995        | WIN |

### gz vs zlib-ng — wall ratio (after/zlib-ng)
| corpus   | gz wall | zlib-ng wall | **gz/zlib-ng** | verdict |
|----------|--------:|-------------:|---------------:|---------|
| **logs** | 170 ms  | 178 ms       | **0.955**      | **WIN** |
| silesia  | 668 ms  | 799 ms       | 0.837          | WIN |
| nasa     | 217 ms  | 292 ms       | 0.744          | WIN |
| monorepo | 109 ms  | 127 ms       | 0.858          | WIN |
| logsbig  | 786 ms  | 860 ms       | 0.914          | WIN |

### THE logs flip number
**logs gz/igzip = 1.029 — the stacked cache+CRC do NOT flip logs to a WIN.** It is a
~3% loss (more than the optimistic ~1.022 the per-lever docs projected; the two levers
were never measured together before this run). gz **does** beat zlib-ng on logs (0.955)
and on every corpus.

**Premise correction (HYPOTHESIS→measured):** zlib-ng is **NOT** the logs leader —
**igzip leads logs** (162 ms < gz 167 ms < zlib-ng 178 ms). The "under-measured
zlib-ng is the logs leader" framing is falsified here; igzip is the cell to beat.

**Cross-corpus dominance CONFIRMED:** gz beats igzip on silesia/nasa/monorepo (and
ties-to-wins logsbig); gz beats zlib-ng everywhere. logs-T1 vs igzip is the ONE
remaining losing cell.

---

## STAGE 3 — remaining logs gap decomposition

logs, cache+CRC stacked (gz perf counters, interleaved):
- gz **1.038 cyc/B**, **instr/B 3.30**, **IPC 3.18** (the highest IPC of any corpus
  bar logsbig).
- igzip **1.013 cyc/B** → remaining gap **~0.025 cyc/B (~2.5%)**, wall ~2.9%.

High IPC + the gap being a cyc/B gap ⇒ gz is **instruction-bound** on logs, not
IPC/stall-bound. With the CRC compute now ~halved (VPCLMULQDQ) and per-block table
rebuilds deduped (cache), the residual instruction surplus is **not** CRC or
table-build.

**HYPOTHESIS (unvalidated — cites prior gated finding `project_copy_floor_instr_locate_2026_06_25`):**
the residual logs gz/igzip surplus is the **back-ref COPY path** (that finding
measured logs gz/igzip instr/B = 1.29×, gz IPC HIGHER than igzip → instruction-COUNT
in the copy path; gz uses scalar 8 B movs vs igzip wide SIMD). The **copy lever
(STACKING-AUDIT #B)** targets exactly this region.

**Does copy (B) finish the flip?** UNPROVEN. The remaining gap is ~2.5% cyc/B and the
copy path is the dominant remaining instruction surplus on copy-heavy logs, so closing
a meaningful fraction of the 1.29× copy surplus is the **candidate** to flip logs to
≤1.0 vs igzip. This requires the copy lever to be built and gated (fulcrum abmeasure,
same env) to confirm — it is NOT established by this run.

---

## Verdict

- **Stage 1 (merge-blocker): PASS** — VPCLMULQDQ fold byte-exact, 857 089 + 4 000 cases, 0 fail.
- **Stage 2:** stacked, gz beats igzip 4/5 and zlib-ng 5/5; **logs-T1 vs igzip stays a ~1.029 LOSS** — cache+CRC do NOT flip the last cell. igzip (not zlib-ng) is the logs leader.
- **Stage 3:** remaining logs gap is ~2.5% cyc/B, instruction-bound; the **copy lever (B)** is the registered next step (HYPOTHESIS) to finish the flip.

**Scope stamp:** Intel i7-13700T x86_64, `stack-cache-crc` @9f83198, N=15, /dev/null,
sha-verified, VOID-QUIET (run-queue 2–3, NOT-YET-LAW). **AMD + aarch64 OWED** — the
VPCLMULQDQ kernel is x86-only (aarch64 takes the validated `fold3` path).
