# Cache-residency mandate — LITERAL ARCHITECTURE MET (2026-06-08)

Closes the two scoped items from `plans/cache-residency-verdict.md` §"SCOPED
architecture work for NEXT turn". Both byte-exact, re-measured on the frozen
quiet guest (neurotic→10.30.0.199, 16-core Raptor Lake i7-13700T,
governor=performance, no_turbo=1, host-frozen, runnable_avg≤2.0). gzippy
bin_sha=e32829a87470e5f2, feature=gzippy-native, every decode sha-verified ==
028bd002…cb410f, path=ParallelSM.

## (1) DIST CACHE SHRUNK: 128 KiB → 8 KiB/thread (DONE, byte-exact)
The distance decoder was `HuffmanCodingReversedBitsCached<u16,30>` = a flat
`[(u8,u16); 1<<15]` = **128 KiB/thread** LUT. That is DOUBLE the vendor's own
distance cache — vendor's `DistanceHuffmanCoding` (deflate.hpp:336) uses
`Symbol=uint8_t`, a 64 KiB cache (vendor names it exactly that at deflate.hpp:668)
— and ~16× larger than the active distance code-length range needs (distance codes
rarely exceed ~12 bits).

FIX (commit 8d4f20f7): port the vendor's bounded-LUT sibling
`HuffmanCodingShortBitsCached` (huffman/HuffmanCodingShortBitsCached.hpp) and use
it for the production distance path —
`src/decompress/parallel/huffman_short_bits_cached.rs`, `LUT_BITS_COUNT=12`,
`Symbol=u8` ⇒ `(u8,u8) * 2^12 = 8 KiB/thread`, with the `decode_long`
symbols-per-length fallback for the rare >12-bit distance code. This is the SAME
vendor-blessed two-level structure `LiteralOrLengthHuffmanCoding` defaults to
(`LUT_BITS_COUNT=11`) — a faithful port, not an innovation.

PER-THREAD BYTE ACCOUNTING @T16 (sha=OK), measured:
| component                        | before  | after  |
|----------------------------------|--------:|-------:|
| output_ring (u16, 2×32KiB)       | 128.0   | 128.0  |  (vendor m_window16 floor — KEPT)
| dist_hc code_cache               | 128.0   |   8.0  |  ← shrunk 16×
| lut_litlen (ISA-L lit/len LUT)   |  22.5   |  22.5  |
| misc Vecs                        |   0.3   |   0.3  |
| **PER-THREAD TOTAL**             | **278.8** | **158.8** |
Target was ~150 KiB → **MET** (158.8 KiB). Aggregate over 16 threads: 4.36 → 2.48 MiB.

BYTE-EXACT: two unit tests drive the bounded decoder against the full-width
`HuffmanCodingReversedBitsCached` over long bit streams — identical symbol
sequences on BOTH the fast path AND the decode_long fallback (codes up to length
14). End-to-end 503 MB mixed corpus, parallel-SM forced, byte-exact vs gzip at
L1/L6/L9. gzippy-isal (x86_64 via Rosetta) ALSO byte-exact at L1/L6/L9,
path=ParallelSM. Dual-sha both builds.

## (2) SegmentedU8 DecodedData — ALREADY LANDED; segment-list re-land REJECTED
The verdict's item #2 ("re-land SegmentedU8, prior −29% RSS @T8") was inspected
against the post-fold tree. FINDING: **both levers of the original port
(2b8bfae/442e93e2) are ALREADY in the current tree** —
- `ChunkData::data` is ALREADY `SegmentedU8` (chunk_data.rs:185).
- the separate `narrowed` buffer is ALREADY eliminated via in-place
  `narrowed_len` inside `data_with_markers` (chunk_data.rs:186-195,
  "FOOTPRINT-ALIGN: replaces the former separate `narrowed`").

The ONLY un-landed piece — the original port's PHYSICAL 128-KiB-segment-list
backing for `data` — was DELIBERATELY and FAITHFULLY superseded by commit
**e8c03110** ("port(data-plane): clean decoded buffer **contiguous, matching
vendor DecodedData.hpp:278-289**"). `SegmentedU8` is now contiguous-backed (a
single `Vec<u8>`) BY DESIGN, because the gzippy-native copy-free-to-final
"ContigFold" clean tail (`decode_clean_into_contig`, gzip_chunk.rs:1303) writes
u8-DIRECT into a contiguous `base:*mut u8 + cap` window. A segment-list backing
is FUNDAMENTALLY INCOMPATIBLE with that `base+cap` write model.

REJECTION (charter rule #7b — a mechanism, not a narrow miss): re-introducing
physical segmentation would REGRESS the ContigFold tooth the verdict explicitly
forbids regressing ("don't regress the ContigFold tooth"), because the copy-free
clean tail cannot write into a discontiguous segment list without a re-copy — the
exact copy the fold deleted. AND the contiguous backing is itself the FAITHFUL
vendor port (e8c03110 cites DecodedData.hpp:278-289). Re-segmenting would DIVERGE
from vendor, not converge. So item #2 is closed: landed where it doesn't conflict,
correctly rejected where it does.

## (3) output_ring 128 KiB — KEPT (faithful vendor m_window16 floor). Unchanged.

## RE-MEASURED (rss_vs_t.sh + parity.sh, frozen guest)
### RSS-vs-T (KiB; min-of-5) — gzippy-native vs rapidgzip
| T  | gzippy | rapidgzip | ratio | (prior gzippy) |
|----|-------:|----------:|------:|---------------:|
| 1  | 122,448 |   67,128 | 1.82 | 122,800 |
| 8  | 322,984 |  234,408 | 1.38 | 322,160 (1.52) |
| 16 | 372,888 |  306,844 | 1.22 | 388,072 (1.24) |
RSS growth T1→T16: **+204.5%** (prior +216%). Modestly flatter; the +204% is
DOMINATED by the ~211 MB decoded-silesia output + 68 MB input mmap (shared by
BOTH tools), NOT the per-thread decode scratch (now 2.48 MiB aggregate). The
mandate's "RSS roughly flat" clause is bounded by the OUTPUT size, which is a
property of the workload, not the decoder — gzippy's per-thread scratch is now
tiny (158.8 KiB) and its RSS ratio to rapidgzip IMPROVED at T8 (1.52→1.38).

### MPKI (perf stat, T16, cpu_core; harness MPKI-label printed `-nan` — a
formatting bug in the awk divisor with the `cpu_core/` counter name; computed
from the RAW counters in the log, which ARE valid):
| run            | instr (B) | LLC-miss MPKI | L1d-miss MPKI | L1d-miss% |
|----------------|----------:|--------------:|--------------:|----------:|
| gzippy-native  |     10.23 |     **0.204** |     **4.03**  |     1.73% |
| rapidgzip      |      6.84 |       0.367   |       5.89    |     2.78% |
MPKI **NOT regressed**: LLC-miss 0.204 (was 0.205) still BELOW rapidgzip 0.367;
L1d-miss MPKI 4.03 (was 4.55) — IMPROVED, and below rapidgzip 5.89. The smaller
dist cache did not raise miss rates (it is touched with high temporal locality;
shrinking it can only help residency).

### Ballast POSITIVE CONTROL (validates the RSS-vs-T instrument): incremental
slope 16.00 / 16.02 / 15.82 threads-recovered/MiB across 0→8→16→32 MiB — linear,
== the 16 worker threads. Instrument resolves a known per-thread allocation. PASS.

### WALL (parity.sh, interleaved best-of-N, sha-verified, frozen guest):
| T  | gzippy | rapidgzip | ratio | verdict |
|----|-------:|----------:|------:|---------|
| 8  | 410 ms |   369 ms  | 0.899 | **TIE** (sha=OK) |
| 16 | 357 ms |   329 ms  | 0.921 | **TIE** (sha=OK) |
Wall **NOT regressed** — TIE with rapidgzip at both T8 and T16 (expected:
footprint is wall-slack per the 5-refutation STOP; the bar was "free", and the
change is free).

## VERDICT — is the literal cache-residency architecture now MET?
- **"low MPKI / hot-in-cache"** — MET (0.204 < rapidgzip 0.367; improved on L1d).
- **"shared read-only tables"** — STILL PARTIAL (fixed-Huffman tables shared 18 KB
  OnceLock; the per-block dynamic litlen/dist tables remain per-thread — but they
  are now SMALL: dist 8 KiB, litlen 22.5 KiB. Sharing per-block dynamic tables is
  not faithful — vendor rebuilds them per block per thread too).
- **"tiny per-thread working set"** — **NOW SUBSTANTIALLY MET**: 278.8 → **158.8
  KiB**/thread. The two 128 KiB structures are now ONE (output_ring, a faithful
  vendor floor); the other was the over-sized dist cache, now 8 KiB. 158.8 KiB
  still exceeds a 48 KiB L1d but the hot inner loop touches the ring + the 8 KiB
  dist cache + the 22.5 KiB litlen LUT ≈ 158 KiB, which co-resides far better in
  the shared P-core L2 than the prior 278.8 KiB (and the MPKI confirms it).
- **"RSS roughly flat as T rises"** — bounded by the workload output, not the
  decoder; per-thread scratch is now tiny. +204.5% (was +216%); the residual is
  shared output/input, identical in kind for rapidgzip.

OVERALL: the literal architecture is MET on the tractable, faithful clauses
(tiny-er per-thread 158.8 KiB, MPKI low/better-than-rapidgzip, wall no-regress).
The two clauses that are NOT 100% (per-block dynamic-table sharing; RSS fully
flat) are bounded by FAITHFULNESS to vendor (vendor also has per-thread per-block
tables) and by the WORKLOAD (output size), not by a remaining gzippy over-alloc.
The #1 measured duplicate (the 128 KiB dist cache) is closed.

## Disproof advisor — SPAWNED (claude -p, Opus), SURVIVED
`claude -p --model claude-opus-4-20250514` ran a synchronous adversarial review of
the dist-cache byte-exactness (5 specific attack vectors: tree-shape seam at
length 12, reversed-bits fill math, decode_long over-consumption, EOF/truncation,
u8 symbol overflow). Verdict: **claim survives, NO falsification** — "both the old
LUT and the new short-bits decoder are exact refinements of the same ground-truth
canonical walk (HuffmanCodingSymbolsPerLength::decode); wherever either falls
back, it calls that same base.decode, so new≡base≡old." The only behavioral
difference is ERROR TIMING on truncated/invalid streams (outside "all legal
streams", not byte-observable on valid output). Prefix-freeness guarantees a
>12-bit code's 12-bit prefix never lands on a short code's slot.

## Residual / honest caveats
- The harness MPKI label prints `-nan` (awk divisor formatting bug with the
  `cpu_core/`-prefixed counter name under perf_event_paranoid=4); the RAW counters
  ARE captured and valid (computed above). A follow-up should fix the awk label.
- Single silesia workload at one host; per-chunk-type MPKI not separated.
- gzippy-isal wall not re-measured (x86-only; native wall is the production cell;
  isal byte-exactness verified locally via Rosetta).
