# Dead End: Page-fault / TLB-walk removal as a wall lever

## Hypothesis

gzippy faults 2.55× more than rapidgzip (223k vs 87k minor-faults, frozen T8-noSMT).
dtlb_store_walk is 3.26× higher. 99.6% of faults are in `worker.decode_chunk`.
Removing this cost (via warm-pool, THP, segmentation, or SLAB retain) would close a
proportional fraction of the 1.47× wall gap.

## How Measured

- **E0 fault-cost amplification** (`GZIPPY_FAULT_AMP=K`): re-faults decode buffers K×
  via madvise(DONTNEED)+re-touch. K=2/4/8/16 → wall +2.6/+23.5/+75.5/+176.7%
  (busy-spin); SURVIVES frequency-neutral SLEEP control (+20/+24/+50/+78%). Faults
  ARE on the critical path when amplified. (Positive control: minor-faults scaled
  linearly 190k→2.47M, dtlb-store-walks 2.55M→41.6M, sha byte-identical every K.)

- **Warm-pool oracle uncap+presize**: dtlb-store-miss 9–15% cut, minor-faults 3–8%
  cut. Wall TIE: T4 –1.7% / T8 –1.1% / T16 +0.0% within stdev 1.9–5.9pp.

- **Never-warmup oracle** (`DECODE_REPEAT=21`): fully-warm passes, faults→~104k (near
  rapidgzip's 87k). T8 –1.43% / T16 +1.52% = SLACK. (N=15 apparent –5.5% blip
  collapsed to –1.4% at N=21 = noise.)

- **SLAB warm-reuse retain** (`GZIPPY_SLAB_ALLOC=1`): wall within noise, sha MATCH.

- **MADV_HUGEPAGE** (`GZIPPY_HUGEPAGE`, `chunk_buffer_pool.rs:113-119`): –38% wall
  regression. Confirmed dead.

## Verdict: REFUTED — page-faults are overlapped SLACK

Amplify-binds / remove-doesn't is real overlapped-slack physics:
- Slowing every worker's fault cost past the consumer's slack propagates head-of-line
  into wall (MAX over a saturating resource) — amplification BINDS.
- Speeding workers just makes them finish earlier and WAIT — slack is WALL-INVISIBLE;
  removal DOESN'T help.

This is Rule 3 (slow-down slope ≠ speed-up ceiling) applied exactly. The E0 positive
control proves criticality (faults are touchable) but does NOT bound the speed-up
(the removal oracle does, and it TIEs).

**Walks=6% of the amplified cost** (decomposing amplifier: zeroing 73%, fault-handler
13%, walk 6%). The 3.26× dtlb_store_walk scare is real TLB pressure but the walks
themselves contribute only 6% when the cost is amplified — too small to explain 1.47×.

## Structural Explanation

Page-faults are a **correlate** of rapidgzip's leaner structure, not an independent
cause. rapidgzip faults 2.55× less because its 128 KiB span-cached buffer lifecycle
naturally produces fewer first-touch events. Killing the correlate (which we did
thoroughly) leaves the 1.47× not unexplained but redirected to the structural
in-order consumer window-resolution chain.

## Code Locations

- `src/decompress/parallel/chunk_buffer_pool.rs` — warm-pool implementation,
  `GZIPPY_HUGEPAGE` gate, `GZIPPY_SLAB_ALLOC` gate
- `src/decompress/parallel/rpmalloc_alloc.rs` — HUGE path (>3.937 MiB munmaps on
  free); MEDIUM/LARGE (≤128 KiB) are span-cached
- `chunk_fetcher.rs` (experiment branches) — `GZIPPY_FAULT_AMP` injection site

## Do Not Re-attempt

- Simple buffer retain (3 oracle TIEs; the oracle was warm-monolith, not
  segmented — see caveat below)
- MADV_HUGEPAGE (–38% regression; walks = 6% of cost so THP attacks the wrong axis)
- Zeroing elimination alone (73% of cost but zeroing is OS-mandatory for first-touch
  security; only bypass = pre-faulted warm pages which already TIE'd)

**Caveat**: the removal oracles warmed the still-MONOLITHIC 12 MiB buffer; they did
NOT test the 128 KiB segmented buffer's TLB residency benefit. The segmentation
lever (`feat/footprint-align`) is separately held/open. The fault-removal refutation
does NOT pre-empt segmentation — see `docs/dead-ends/footprint-align-segmented.md`.

## Related Entries

- `docs/dead-ends/footprint-bandwidth.md` — broader footprint/DRAM theory (6 refs)
- `docs/dead-ends/footprint-align-segmented.md` — the segmentation lever (held)
- `project_pagewalk_faults_are_slack_2026_06_01` memory — airtight advisor-confirmed
  verdict with full oracle accounting
