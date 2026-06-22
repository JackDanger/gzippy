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

## OWED (next, this cycle)
- Clean per-chunk removal-oracles (code change, byte-transparent, non-inert counter each):
  recycle output buffer (#1), reuse window tail buffers (#2), and a boundary-record no-op
  stub (#4) — in `drive_thin_t1_oracle` / `decode_chunk`, gated on T==1.
- AMD/Zen2 replication of the headline (this same script, freeze/thaw).
- Native-kernel race on the real path (gzippy-native `prod` vs igzip), both arches.
- T>1 load-bearing classification for each real T1 cost (T4/T8 must not regress vs rg).
