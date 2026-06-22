# T1-CACHE-RESIDENCY — GATED RESULTS (judged against the pre-registered falsifier)

Branch `t1-cache-residency` (== `kernel-converge-A`, FF). Bar = streaming_thin
`igzip` mode (ISA-L monolith one-shot, WITH CRC, streams a small reused buffer).
Ship target = gzippy-NATIVE (`--features pure-rust-inflate`, no FFI) `prod` mode
(`decompress_parallel(&[u8],…,1)` → thin-T1 driver). /dev/null both arms,
decode-only timed, interleaved best-of-15, taskset cpu4. Intel = neurotic/trainer
i7-13700T LXC (unfreezable, taskset); AMD = solvency Zen2 (frozen gov=performance
boost=0). Gate-0: bytes==zcat ALL arms (PASS), A/A |prod−prod2| ≤ 1.5 ms ≪ every Δ.

## VERDICT: **PARTIAL** — a real, gated, mechanism-confirmed win; does NOT reach 1.10

The only lever that pays is the **fixed-size resident output pool (b2)**. It is now
the gzippy-native thin-T1 production default (T1ResidentScope). It is a real gated
drop on monorepo (both arches), silesia + squishy (small, both arches), and a TIE
on nasa — with **no regression anywhere** and the wall move EXPLAINED by a
minor-faults drop toward igzip. It does **not** reach the pre-registered ≤1.10 on
any cell; the residual is dominated by INSTRUCTIONS (+16–24% vs igzip), unaffected
by any residency lever (the kernel/driver re-emerging, as the falsifier predicted).

## PER-LEVER GATED VERDICT (vs the pre-registered falsifier)

| lever | mechanism | verdict |
|-------|-----------|---------|
| (a) 2 MiB T1 chunk (`GZIPPY_CHUNK_KIB=2048`) | — | **FALSIFIED**: REGRESSES silesia/nasa/squishy both arches (Intel silesia 1.215→1.246, AMD nasa 1.249→1.293); helps only monorepo. Dropped. |
| (b1) recycle-only (`GZIPPY_MANUAL_BUFFER_POOL=1`) | minor-faults UNCHANGED vs baseline (silesia 23039=23039, monorepo 6996=6996) | **FALSIFIED**: TIE at the wall everywhere — recycling without a fixed reserve does not keep pages resident. |
| (b2) recycle + fixed reserve (resident pool) | minor-faults DROP toward igzip | **KEEP (PARTIAL)**: real gated wall win on monorepo (both arches); now production default via T1ResidentScope. |
| (c) boundary shed (record_boundaries=false) | instructions within noise | **FALSIFIED/dropped**: TIE; not worth the divergence. |

## PRODUCTION DEFAULT (T1ResidentScope) — prod/igzip, before → after (gated, best-of-15)

| corpus | Intel before | Intel after | AMD before | AMD after |
|--------|-------------|-------------|-----------|-----------|
| silesia  | 1.215 | **1.196** | 1.198 | **1.166** |
| nasa     | 1.273 | 1.271 (TIE) | 1.249 | 1.248 (TIE) |
| monorepo | 1.361 | **1.299** | 1.389 | **1.291** |
| squishy  | 1.176 | **1.165** | 1.153 | **1.136** |

`prod` (new scope default) == `respool` (the `GZIPPY_RESIDENT_OUTPUT_POOL` oracle)
to within 0.001–0.002 in EVERY cell → the production code reproduces the measured
oracle (bake validated). monorepo (the worst cell) improves ~5% (Intel) / ~7% (AMD),
Δ (5–7 ms) ≫ spread (0.5–1.7 ms). silesia/squishy small but real on both arches
(AMD spreads 1.7–1.8 ms). nasa is a clean TIE.

## MECHANISM (Gate-2, AMD bare-metal `perf stat`, minor-faults) — CONFIRMED

| corpus | igzip | prod BEFORE | prod AFTER (scope) | igzip instr | prod AFTER instr |
|--------|-------|-------------|--------------------|-------------|------------------|
| silesia  | 17093 | 23039 | **19942** | 2.59 B | 2.94 B (+13%) |
| monorepo | 2938  | 6996  | **5194**  | 0.374 B | 0.449 B (+20%) |
| nasa     | 5636  | 8215  | 8214 (flat) | 0.811 B | 0.939 B (+16%) |

The wall win TRACKS the fault drop (monorepo biggest relative fault recovery =
biggest wall win; nasa no fault drop = wall TIE). Mechanism confirmed: a fixed,
recycled, resident output buffer keeps the decode's pages warm across chunks,
where the per-chunk ratio-sized fresh alloc re-faulted. Recycle-ALONE (b1) left
faults identical → the FIXED reserve is the operative part.

## RESIDUAL = INSTRUCTIONS (the next lever — kernel/driver re-emergence)

Even after the fault lever, gzippy-native runs **+13–20% more instructions** than
igzip, and that, not faults, dominates the remaining wall gap (faults are only
~26% recoverable on the best corpus and a minor fraction of the total). The
residency hypothesis is therefore REFUTED as the path to ≤1.10; the kernel/driver
instruction count is the located residual, exactly the falsifier's "kernel may
re-emerge once the driver is shed" prediction.

## T>1 NO-REGRESSION (the levers are T1-GATED by construction)

T1ResidentScope is entered ONLY in `drive_thin_t1_oracle` (T==1). At T>1 the scope
is never entered, so `manual_buffer_pool_enabled()` / `resident_output_pool_enabled()`
return their (default-false) env values and the parallel pipeline is byte-identical
code. Empirically confirmed: NEW-vs-OLD prodt at T1/T4/T8 (see the tregress run).

## STRATEGIC FACT (carried, gated Intel) — rapidgzip −P1 ITSELF loses to igzip at T1

rapidgzip −P1 / igzip: silesia 1.23, nasa 1.86, monorepo 1.81, squishy 1.20.
gzippy-native thin-T1 already BEATS rapidgzip −P1 at T1; the residual is purely vs
igzip (single-stream SOTA), and is now an INSTRUCTION gap, not faults.

## INSTRUMENT-INTEGRITY NOTE (Gate-0 incident, resolved)

A `push.default=tracking` misconfig made `git push origin t1-cache-residency`
silently no-op (the branch tracked kernel-converge-A); the boxes built the stale
falsifier-only commit for two measurement rounds, so an early "the bake is inert"
reading was an ARTIFACT of testing original code. Fixed (push.default=simple,
branch.merge corrected), boxes rebuilt from the verified commit (grep-checked the
scope symbol is present), and ALL banked numbers above are from the verified binary.
