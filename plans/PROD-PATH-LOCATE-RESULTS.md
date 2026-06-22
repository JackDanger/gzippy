# PROD-PATH LOCATE (MEASUREMENT 4) — the REAL production T1 path vs igzip

Branch `kernel-converge-A`. Instrument: `examples/streaming_thin.rs` **`prod`** mode
(= `single_member::decompress_parallel(&[u8], …, 1)`, the real production T1 driver,
borrowing the input — NOT the `cheap` proxy that added an input `.to_vec()` copy) +
`scripts/bench/_prod_path_locate_guest.sh`. Build = **gzippy-isal**, so every T1 chunk's
clean tail decodes through **REAL ISA-L `_04`** (`isal_clean_tail` cfg) — inner kernel
IDENTICAL to igzip — so `prod − igzip` IS the production DRIVER/SCAFFOLD overhead.
Both arms `/dev/null`, file pre-loaded, decode-only timed, interleaved best-of-15, pin cpu4.

Gate-0: every arm bytes==zcat (PASS all cells). A/A `|prod − prod2|` reported per cell
(≪ inter-arm Δ). Routing (Gate-4): `prod` fires `thin-T1 … marker_chunks=0` (the real
`drive_thin_t1_oracle` spine) and the clean tail is real ISA-L (`isal_clean_tail` build).
Non-inert proofs (GZIPPY_DEBUG, captured): `prodcrcoff`→"ORACLE CRC_OFF ACTIVE"; `prodbig`
→ stride 1024KiB→22528KiB; `prodpool`/`prod`→thin-T1.

Arms: `igzip` (ISA-L monolith WITH CRC = bar) · `prod` (real prod T1, CRC on) · `prod2`
(A/A) · `prodcrcoff` (GZIPPY_ORACLE_CRC_OFF=1, removes #5 CRC 2nd-touch) · `prodbig`
(GZIPPY_CHUNK_KIB=huge, collapses per-chunk count) · `prodpool` (GZIPPY_RESIDENT_OUTPUT_POOL=1, #1 reserve pin).

## HEADLINE — the proxy MISSED the real scaffold

The prior SCAFFOLD-LOCATE measured the `cheap` PROXY and concluded "no scaffold to
converge; the contig driver meets-or-beats igzip once the harness input copy is excluded."
That was a **proxy artifact**: the proxy OMITTED the real production driver. On the REAL
production path the driver is **28–35% slower than the igzip monolith** (Intel), and only
2–8% of that is CRC. There is a large (~20–28%) production-only overhead the proxy never saw.

## INTEL (neurotic i7-13700T LXC, taskset cpu4, best-of-15, load ~3-4)

| corpus   | igzip ms | prod ms | **REAL SCAFFOLD** (prod−igzip)/igzip | CRC share (prod−crcoff)/prod | SCAFFOLD−CRC (crcoff−igzip)/igzip | reserve#1 (prod−pool)/prod | A/A ms |
|----------|----------|---------|--------------------------------------|------------------------------|-----------------------------------|----------------------------|--------|
| silesia  | 655.3    | 864.9   | **+31.98%** (209.6 ms)               | 2.34% (20.2 ms)              | 28.90%                            | 1.72% (14.9 ms)            | 0.43   |
| nasa     | 232.2    | 299.1   | **+28.79%** (66.8 ms)                | 7.77% (23.2 ms)              | 18.79%                            | 0.25% (0.7 ms)             | 0.05   |
| monorepo | 105.5    | 142.7   | **+35.21%** (37.2 ms)                | 3.44% (4.9 ms)               | 30.55%                            | 5.49% (7.8 ms)             | 0.39   |
| squishy  | 1288.5   | 1672.5  | **+29.80%** (384.0 ms)               | 2.00% (33.4 ms)              | 27.20%                            | 0.17% (2.9 ms)             | 6.39   |

A/A ≤ 6.4 ms ≪ every reported Δ (silesia 209 / nasa 67 / monorepo 37 / squishy 384).
GATE0(bytes)=PASS all.

### CONFOUNDED arm — do NOT bank `prodbig` per-chunk-family as clean
`prodbig` (GZIPPY_CHUNK_KIB=huge) gave silesia +4.38% / nasa **−21.37%** / monorepo +9.47%
/ squishy −2.25%. The sign-inconsistency (nasa much SLOWER with fewer chunks) shows the
knob is CONFOUNDED: collapsing chunk COUNT also balloons each chunk's `compute_initial_reserve`
(= compressed_span × factor, 64MiB cap) and ISA-L decode-region size, so a ~68MiB-output
chunk over-reserves + reallocs + first-touch-faults a huge buffer at once. It bundles
chunk-count with per-chunk-reserve-size → NOT a clean per-chunk removal-oracle. The clean
per-chunk-cost oracle (recycle the output buffer, byte-transparent) is OWED (code change).

### `prodpool` (#1 reserve pin) is a WEAK oracle
`drive_thin_t1_oracle` calls `ChunkData::new` (no recycler), so RESIDENT_OUTPUT_POOL only
pins reserve SIZE to 64MiB without recycling the buffer across chunks — it does not test
the first-touch/alloc cost cleanly. Small/noisy result (0.17–5.49%). The clean #1 oracle =
recycle the ChunkData output buffer in the thin-T1 loop (OWED, code change).

## GATED FINDINGS (Intel; removal-oracle = verdict)
1. **REAL production T1 driver overhead vs igzip = +28.8 to +35.2%** on all 4 corpora —
   roughly 4–5× the proxy's 1.5–10.7% and POSITIVE everywhere (vs the proxy's "gz beats
   igzip"). The proxy `cheap` arm was NOT the production driver; this is.
2. **CRC32 second-touch (#5) = 2.0–7.8%** of prod wall (clean removal, non-inert proven).
   Largest on nasa (7.8%, smallest output-relative-to-time corpus). Real but minor.
3. **SCAFFOLD−CRC = 18.8–30.6%** remains AFTER CRC removed — the dominant production-only
   overhead is NOT CRC. It must be the per-chunk data-plane (#1 alloc/first-touch / #2
   window-roll / #3 isal lifecycle / #4 boundary record). Clean per-candidate oracles OWED.

## AMD (solvency EPYC 7282 Zen2, bare-metal FROZEN gov=performance boost=0, taskset cpu4, best-of-15)
Box thawed + verified after: gov=ondemand / boost=1 / paranoid=4.

| corpus   | igzip ms | prod ms | **REAL SCAFFOLD** (prod−igzip)/igzip | CRC share (prod−crcoff)/prod | SCAFFOLD−CRC (crcoff−igzip)/igzip | reserve#1 (prod−pool)/prod | A/A ms |
|----------|----------|---------|--------------------------------------|------------------------------|-----------------------------------|----------------------------|--------|
| silesia  | 357.4    | 472.8   | **+32.31%** (115.5 ms)               | 4.31% (20.4 ms)              | 26.61%                            | 3.07% (14.5 ms)            | 0.35   |
| nasa     | 131.2    | 169.2   | **+28.94%** (38.0 ms)                | 11.99% (20.3 ms)             | 13.49%                            | 0.33% (0.6 ms)             | 1.54   |
| monorepo | 57.9     | 82.7    | **+42.82%** (24.8 ms)                | 5.37% (4.4 ms)               | 35.15%                            | 8.40% (6.9 ms)             | 0.11   |
| squishy  | 703.8    | 906.4   | **+28.79%** (202.6 ms)               | 4.12% (37.3 ms)              | 23.49%                            | 0.89% (8.1 ms)             | 1.12   |

A/A ≤ 1.5 ms ≪ every reported Δ. GATE0(bytes)=PASS all. `prodbig` confounded again
(nasa −46.4% / squishy −16.7% — same reserve-balloon confound as Intel; do NOT bank).

## CROSS-ARCH VERDICT (Intel AND AMD AGREE → LAW-grade)
1. **The REAL production T1 driver is +28.8 to +42.8% slower than the igzip monolith on
   BOTH arches** (Intel 28.8–35.2%, AMD 28.9–42.8%), positive on all 8 cells, holding the
   inner kernel constant (isal clean tail = igzip `_04`). This OVERTURNS the prior
   SCAFFOLD-LOCATE "no scaffold to converge" verdict — which was a PROXY artifact (the
   `cheap` arm omitted the real production driver). There IS a large real scaffold.
2. **CRC32 second-touch (#5) = 2.0–12.0%** of prod wall, both arches (larger on Zen2 +
   on nasa). Real, clean removal-oracle (non-inert proven), but a minority of the gap.
3. **SCAFFOLD−CRC = 13.5–35.2%** remains after CRC removed, both arches — the DOMINANT
   production-only overhead is the per-chunk data-plane, NOT CRC. Clean per-candidate
   removal-oracles (#1 alloc/first-touch, #2 window-roll, #3 isal lifecycle, #4 boundary
   record) are OWED — `prodbig`/`prodpool` could not isolate them (confounded/weak).

## PER-CHUNK FIXED COST — chunk-size sweep (the CLEAN per-chunk oracle)

`prodbig` (CHUNK_KIB=huge) was confounded because a giant chunk balloons each chunk's
`compute_initial_reserve` past the 64MiB cap + reallocs. The CLEAN per-chunk-count
perturbation is a chunk-SIZE SWEEP within the sane reserve range (256KiB–4MiB, all under
the cap): varying chunk count ~8× while reserve stays proportional. `prod` mode + real
ISA-L tail, GZIPPY_CHUNK_KIB ∈ {256,512,1024,2048,4096}, igzip bar, best-of-11, pin cpu4.
Non-inert: GZIPPY_DEBUG `stride=` tracks the knob.

### INTEL (neurotic, pin cpu4, best-of-11) — wall ms by chunk KiB
| corpus   | igzip | 256K  | 512K  | 1024K(default) | 2048K | 4096K | slope 256K→2M | 4M floor vs igzip |
|----------|-------|-------|-------|----------------|-------|-------|---------------|-------------------|
| nasa     | 232.4 | 408.8 | 346.5 | 297.5          | 279.3 | 281.7 | −129.5 ms (−32%) | +21.5%         |
| silesia  | 657.0 | 1321.7| 1013.6| 867.6          | 813.3 | 816.4 | −508.3 ms (−38%) | +24.3%         |
| monorepo | 105.8 | 190.6 | 162.6 | 143.5          | 128.7 | 131.0 | −61.8 ms (−32%)  | +23.8%         |

GATED FINDING (Intel, Gate-2 causal perturbation — monotonic, 8× chunk-count swing):
1. **Per-chunk FIXED cost is the DOMINANT residual lever.** Wall scales with chunk COUNT;
   reducing it (bigger chunks) recovers a LARGE fraction: the 256K→2M slope is −32 to −38%
   of wall. This is the combined per-chunk family (#1 alloc/first-touch + #2 window-roll +
   #3 isal lifecycle + #4 boundary record) — they all fire per chunk.
2. **The T1 production default (1 MiB) is SUBOPTIMAL.** The optimum is ~2 MiB; moving
   1MiB→2MiB recovers nasa −6.4% / silesia −6.3% / monorepo −10.4% with zero code beyond
   the existing knob. (T1_TARGET_COMPRESSED_CHUNK_BYTES=1MiB — re-evaluate; it was chosen
   for "warm output-buffer recycling," not gated on this real-path sweep.)
3. **Irreducible floor ≈ +21–24% over igzip** even at the optimal chunk size — the per-byte
   + per-chunk-irreducible residual (CRC second-touch ~2–8%, segment commit, the single
   window handoff). This is what a structural convergence (not just chunk sizing) must close.
4. **EXPLAINS the prodbig confound:** beyond ~4 MiB the reserve-balloon/realloc/fault cost
   reverses the gain (giant chunk SLOWER), so the curve is U-shaped with a 2–4 MiB optimum.

Mechanism cross-check (neurotic perf, software counter reliable; LXC HW counters
`<not supported>`): prod minor-faults ≈ 1.5× igzip (nasa 8804 vs 5636; silesia 25390 vs
17099) — consistent with the documented 40%-vs-17% page-fault gap, but extra faults
(~13 MB first-touch) account for only ~20% of the wall gap; the rest is per-chunk
INSTRUCTIONS (bookkeeping). Clean HW instruction/cycle counts owed from AMD bare metal.

### AMD (solvency Zen2, FROZEN gov=performance boost=0, pin cpu4, best-of-11; thawed+verified after)
| corpus   | igzip | 256K  | 512K  | 1024K(default) | 2048K | 4096K | slope 256K→2M | 2M floor vs igzip |
|----------|-------|-------|-------|----------------|-------|-------|---------------|-------------------|
| nasa     | 158.9 | 272.2 | 236.8 | 209.3          | 201.5 | 207.4 | −70.7 ms (−26%)  | +27%           |
| silesia  | 357.4 | 704.3 | 547.1 | 475.7          | 456.3 | 473.6 | −248.0 ms (−35%) | +28%           |
| monorepo | 58.1  | 105.1 | 91.3  | 83.3           | 74.6  | 79.5  | −30.5 ms (−28%)  | +28%           |

Same U-shape + 2 MiB optimum as Intel → CROSS-ARCH. 1MiB→2MiB recovers nasa −3.7% /
silesia −4.1% / monorepo −10.4%.

### Mechanism decomposition (AMD bare-metal perf, clean HW counters)
| arm (nasa)  | minor-faults | instructions | cycles | vs igzip |
|-------------|--------------|--------------|--------|----------|
| igzip       | 5,637        | 789.1 M      | 429.1 M| —        |
| prod 1024K  | 8,800        | 950.8 M      | 541.3 M| +20.5% ins / +26% cyc / +56% flt |
| prod 4096K  | 16,515       | 863.7 M      | 551.2 M| +9.5% ins / +28% cyc / +193% flt |
| silesia igzip   | 17,094  | 2514.6 M     | 1187.1 M| —       |
| silesia prod1024| 25,385  | 3138.9 M     | 1520.4 M| +24.8% ins / +28% cyc |
| silesia prod4096| 41,658  | 2823.7 M     | 1549.8 M| +12.3% ins / +31% cyc |

**The U-shape is an instruction↔fault tradeoff:** bigger chunks CUT per-chunk-bookkeeping
INSTRUCTIONS (nasa 951M→864M, −87M; the surplus over igzip halves +20.5%→+9.5%) but
BALLOON first-touch FAULTS (bigger per-chunk reserve, 8.8K→16.5K). 2 MiB balances them.
At the optimum the residual is ~half per-chunk-instructions (the bookkeeping #2/#3/#4
that the chunk-sweep collapses) + ~half per-byte (CRC second-touch + SegmentedU8 commit +
first-touch faults). NOT one clean lever — a real structural convergence (fewer per-chunk
ops AND a recycled/resident output buffer) is needed to fully close it.

### CROSS-ARCH PER-CHUNK VERDICT (Intel + AMD agree → LAW-grade)
- Per-chunk fixed cost is the dominant residual on BOTH arches; wall is U-shaped in chunk
  size with a 2 MiB optimum; the 256K→2M slope is −26 to −38% of wall.
- The T1 production default (T1_TARGET_COMPRESSED_CHUNK_BYTES = 1 MiB) is SUBOPTIMAL on
  both arches; 2 MiB is the gated optimum (cheap immediate win, existing knob).
- Irreducible floor +21–28% over igzip at the optimum = per-chunk-instruction-irreducible
  + per-byte (CRC + commit + faults).

## NATIVE KERNEL ON THE REAL PATH — the kernel is NOT a lever here

Race: `prod_native` (gzippy-NATIVE build, pure-Rust `decode_clean_into_contig` kernel) vs
`prod_isal` (gzippy-isal build, ISA-L `_04` tail) vs `igzip`, SAME production driver
(`decompress_parallel,1`), chunk=1024 default, best-of-11, pin cpu4. `(prod_native −
prod_isal)/prod_isal` = the KERNEL delta on the real path (driver held constant). Both
builds VERIFY OK (byte-exact).

| corpus   | igzip | prod_isal | prod_native | **kernel (nat−isal)/isal** | native_vs_igz |
|----------|-------|-----------|-------------|----------------------------|---------------|
| **Intel (neurotic)** |||||
| nasa     | 232.6 | 297.4     | 302.5       | +1.72%                     | +30.1%        |
| silesia  | 657.1 | 865.4     | 806.3       | **−6.83%** (native faster) | +22.7%        |
| monorepo | 105.7 | 142.5     | 146.2       | +2.59%                     | +38.4%        |
| squishy  | 1287.6| 1677.5    | 1532.5      | **−8.64%** (native faster) | +19.0%        |
| **AMD (solvency, frozen→thawed)** |||||
| nasa     | 130.8 | 167.5     | 163.5       | −2.40%                     | +25.0%        |
| silesia  | 357.2 | 473.5     | 428.3       | **−9.54%** (native faster) | +19.9%        |
| monorepo | 58.0  | 83.3      | 81.3        | −2.38%                     | +40.3%        |
| squishy  | 703.4 | 905.7     | 814.5       | **−10.07%** (native faster)| +15.8%        |

### CROSS-ARCH KERNEL VERDICT (Intel + AMD agree → LAW-grade) — REFRAMES the campaign
1. **On the REAL production path the native pure-Rust kernel is at PARITY-or-FASTER than
   the ISA-L kernel** — kernel delta within ±2.6% on nasa/monorepo and NEGATIVE (native
   FASTER) by 6.8–10.1% on silesia/squishy, on BOTH arches.
2. **The prior decomposition's "Intel kernel ceiling +14.5% (nasa) / +13.8% (monorepo)"
   does NOT survive the real path.** It was a `thin`-PROXY artifact: in the proxy the
   kernel was a large fraction of a near-monolith driver; in the real per-chunk driver the
   per-chunk overhead dominates and MASKS the kernel difference. The heroic inner-kernel
   rewrite the prior cycles debated would buy ~nothing on the real production T1 path.
3. **The ship target (gzippy-NATIVE, no FFI) is +15.8 to +40.3% over igzip — and that gap
   is ALMOST ENTIRELY the DRIVER (per-chunk), not the kernel.** Closing the per-chunk
   driver overhead (chunk sizing + recycled output buffer + shedding per-chunk bookkeeping)
   would bring gzippy-native to near-igzip parity WITHOUT an ISA-L dependency or a kernel
   rewrite. **The lever is the driver, not the kernel.**

## OWED (next, this cycle)
- Clean per-chunk removal-oracles (code change, byte-transparent, non-inert counter each):
  recycle output buffer (#1), reuse window tail buffers (#2), and a boundary-record no-op
  stub (#4) — in `drive_thin_t1_oracle` / `decode_chunk`, gated on T==1.
- AMD/Zen2 replication of the headline (this same script, freeze/thaw).
- Native-kernel race on the real path (gzippy-native `prod` vs igzip), both arches.
- T>1 load-bearing classification for each real T1 cost (T4/T8 must not regress vs rg).
