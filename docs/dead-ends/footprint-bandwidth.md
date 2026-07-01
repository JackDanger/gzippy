# Dead End: Resident-footprint / DRAM-bandwidth as the wall lever

## Hypothesis

gzippy's larger RSS / higher DRAM traffic is the primary cause of the 1.47× T8-noSMT
wall gap vs rapidgzip. Reducing the memory footprint (segment buffers, shrink
allocations, warm-reuse) would close a proportional fraction of the wall.

Motivation: three-way perf analysis 2026-05-28 showed gzippy at parity instructions
(0.98×) but 2.15× more page faults and IPC 1.08 vs 1.42 — pointing at memory stalls.
The alloc+S3 run 2026-06-01 confirmed dtlb_store_walk 3.26× and memory_bound 26.3%
vs 14.8%.

## How Measured — six independent refutations

1. **PR#123 aggregate removal** (`feat/dataplane-2touch`): 128 KiB granule +
   in-place-resolve + writev simultaneously. RSS –20%, store-walks +22%. Wall
   REGRESSED +6–12% T4/T8/T16. The holistic footprint port made things worse.
   (2026-06-02, frozen-clock harness, sha-verified.)

2. **`feat/footprint-align` SegmentedU8** (commit `2b8bfae`): faithful DecodedData
   port, RSS –29% toward rapidgzip. T4/T8 TIE; T16 REGRESSED +5.5% (A3-removal
   mechanism, named). Footprint win does not translate to wall at T4/T8.

3. **Whole-footprint ceiling oracle** (PR#123 variant): shrank maxrss 1040→838 MB
   (30% toward rapidgzip's ~350 MB). Recovered only +3% IPC vs the needed +42%.
   N=21 frozen control REVERSED an apparent N=11 win; minflt UP. Resident footprint
   is overlapped slack.

4. **SLAB warm-reuse retain oracle** (`GZIPPY_SLAB_ALLOC=1`): retained all chunk
   allocations across runs (preventing munmap). Minor faults 218k→223k (flat), wall
   within noise, sha MATCH. Retaining the allocation without segmenting = TIE: the
   first-touch zeroing cost is slack even when the allocation is warm.

5. **MADV_HUGEPAGE falsified** (`GZIPPY_HUGEPAGE`, `chunk_buffer_pool.rs:113-119`):
   measured –38% wall regression. The fault-cost amplifier sub-split confirmed walks
   are only 6% of the amplified cost; zeroing is 73%. THP attacks the wrong axis and
   introduces its own overhead.

6. **Warm-pool never-free oracle** (`DECODE_REPEAT=N`): forced N=21 warm passes so
   faults collapsed to ~104k (near rapidgzip's 87k). Wall TIE at T4 –1.7% / T8
   –1.1% / T16 +0.0% within stdev 1.9–5.9pp. The true warm-reuse ceiling is 0%.

## Verdict: REFUTED

The 3.26× dtlb_store_walk / 2.55× fault gap is a **CORRELATE** of rapidgzip's leaner
structure, not an independent lever. When the cost is amplified (E0, K×), the wall
moves — but amplify-binds while remove-doesn't is the signature of overlapped slack
(slowing every worker forces head-of-line; speeding them just makes workers finish
earlier and wait). Rule 3 (slow-down slope ≠ speed-up ceiling) applies exactly here.

gzippy's instructions are EQUAL to rapidgzip's (10.90B vs 10.84B = 1.01×). The gap
is pure stalls, and those stalls are fundamental in-order-consumer coordination, not
recoverable memory traffic.

## Code Locations

- `src/decompress/parallel/segmented_buffer.rs` — SegmentedU8 scaffold (unwired into
  production `ChunkData::data`)
- `src/decompress/parallel/chunk_buffer_pool.rs` — per-worker pool (MAX_POOLED=8),
  `GZIPPY_HUGEPAGE` gate (lines 113-119, 224-228), `GZIPPY_SLAB_ALLOC` gate
- `src/decompress/parallel/rpmalloc_alloc.rs` — typed rpmalloc path (hot U8/U16
  buffers bypass `#[global_allocator]`; HUGE path >3.937 MiB munmaps on free)

## Do Not Re-attempt

Do not re-attempt any of these without a new mechanism that explains why the prior
oracles were wrong:
- Buffer retain / warm-pool (3 oracles all TIE)
- MADV_HUGEPAGE (–38% regression, walks = 6% of cost)
- Holistic footprint shrink (PR#123 REGRESSED, N=21 oracle reversed)

Segmentation alone (`feat/footprint-align`) is not refuted at T4/T8 but is HELD on
the T16 A3-removal regression. See `docs/dead-ends/footprint-align-segmented.md` and
`docs/open-candidates.md` for the re-entry path.
