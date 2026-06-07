# Parity-final — gzippy ↔ rapidgzip parallel single-member (FAITHFUL pattern parity)

Done-bar (user-set): reach FAITHFUL PATTERN parity even if gzippy stays SLOW
because of its mandated pure-Rust DEFLATE engine. This document is the divergence
LEDGER: every runtime-pattern item is either MATCHED (faithful port) or FIXED
(transliterated this arc). The remaining wall gap is the accepted pure-Rust
engine-speed residual, not a shape/keying/wrapper deviation.

Branch `reimplement-isa-l`. #B transliteration commit `0a40d5e` (rebased onto the clean base after the 3 TEMP instrument commits were dropped; was `7f8dfc8` pre-rebase). Decompressed silesia output sha256 `028bd002c89c9a909ccdbc2af0a223de285348edb014ccc8e27d297f52cb410f` == reference (x86 Rosetta, path=IsalParallelSM) and final locked Fulcrum `RUN_TRUSTWORTHY=true diverged=0`.
Final locked Fulcrum: `/tmp/gzippy-parity-final/fulcrum-report.txt`
(T8, host-locked on neurotic guest, no_turbo + performance governor + frequency
GATE PASS, N=9 interleaved, sha-verified).

---

## Divergence ledger

| # | item | gzippy ↔ vendor | verdict | evidence |
|---|------|-----------------|---------|----------|
| A | two-engine clean-tail handoff | `gzip_chunk.rs:354-561,:773,:1191-1202` ↔ `GzipChunk.hpp:432-526` (`:520-525` hands the ≥32 KiB clean tail to `IsalInflateWrapper` on the WITH_ISAL build) | **MATCHED** | vendor `worker.isal_stream_inflate` busy = 1289.9ms (non-zero) — the handoff exists in BOTH; same `#ifdef WITH_ISAL` selects it and `HuffmanCodingISAL` (#E) |
| B | distance coding | production decode swapped to `HuffmanCodingReversedBitsCached` (was ISA-L `IsalDistCodePure`) ↔ `deflate.hpp:336` (ISA-L distance rejected at `:338`) | **FIXED** (this arc, `0a40d5e`) | ISA-L distance type deleted from `marker_inflate.rs` production path; mirrors canonical reference build `:1482-1484` / decode `:1548`; output bytes UNCHANGED |
| C | window-map lookup keying | resolve-ahead keys `max_acceptable_start_bit` = real decode boundary (`chunk_fetcher.rs:2396,:3230`); consumer keys block-finder real boundary (`:1049,:1410`) ↔ `GzipChunkFetcher.hpp:544` `get(encodedOffsetInBits)` (chunk's OWN field), `:588-590,:712` | **MATCHED (faithful-accept)** | vendor lookup is on the chunk's own field, NOT the sorted-predecessor's published end (`:557` is a publish key, not a lookup); an attempted "faithful re-key" CRC32-BREAKS at T=8 because gzippy's speculative cache is keyed on a partition GUESS pre-block-finder-confirmation — no confirmed boundary exists to key on (`plans/resolve-ahead-rejection.md`). Keeps the faithful 90.2% window-absent fraction. |
| D | marker resolve per-link | decode pri 0 (`chunk_fetcher.rs:1957/1961`) / resolve pri -1 on pool (`:2104`); single-pass fused 64 KiB LUT (`segmented_markers.rs:461-497`) ↔ decode pri 0 / `submitTaskWithHighPriority` pri -1 (`BlockFetcher.hpp:610`, `GzipChunkFetcher.hpp:579`); LUT applyWindow (`DecodedData.hpp:315-338`, which also deliberately avoids `std::transform`) | **MATCHED (faithful-accept)** | placement exact; gzippy resolve SELF-TIME 850ms < rapidgzip 16123ms (gzippy is FASTER here); the 251ms wall-critical in this role is consumer WAIT on the slower pure-Rust decode = engine residual, not a resolve-shape divergence |
| E | lit/len Huffman | `IsalLitLenCodePure` TRIPLE_SYM (`isal_huffman_pure.rs`) ↔ `HuffmanCodingISAL.hpp` | **MATCHED** | faithful multi-symbol port; both `#ifdef WITH_ISAL` |
| F | known-window chunk via inflate wrapper | `gzip_chunk.rs:309` ↔ `GzipChunk.hpp:192` | **MATCHED** | both use the stream engine for known-window chunks |
| G | Block-internal u16→u8 flip predicate | `marker_inflate.rs` flip ↔ `deflate.hpp:1283-1287` | **MATCHED** | byte-for-byte port; fires at runtime (`flip_to_clean=29`) |

**No remaining open pattern divergence.** A was the inverted item (now MATCHED);
B is FIXED; C/D are faithful-accept (independently advisor-corroborated this arc);
E/F/G were already MATCHED.

---

## Sha-verified wall (final Fulcrum, T8)

| | gzippy (pure-Rust) | rapidgzip (ISA-L) | ratio |
|---|---|---|---|
| wall | 841.1ms | 460.6ms | **1.83×** |
| d_c (clean decode/chunk) | 85.5ms | 44.1ms | 1.94× |
| d_w (window-absent decode/chunk) | 122.5ms | 66.7ms | 1.84× |
| L_resolve (mean/link) | 19.94ms | 10.72ms | 1.86× |

Output byte-exact: silesia decompressed sha256
`028bd002c89c9a909ccdbc2af0a223de285348edb014ccc8e27d297f52cb410f` (the reference)
via `path=IsalParallelSM`; lib suites routing 43/0, correctness 140/0,
pure_rust_inflate_corpus 5/0 (incl. silesia CRC stress + corpus + resumable-real).

The d_c/d_w/L_resolve ratios are UNIFORM ~1.85–1.94× — the signature of a single
cause (the pure-Rust-vs-ISA-L engine substitution), not a localized shape gap.

---

## Window-absent fraction — faithful

RUNTIME window-absent **90.2%** (causal §[1]) vs STATIC boundary fraction 31.0%.
gzippy goes window-absent MORE than the layout forces — faithfully matching the
rapidgzip pattern (rapidgzip runtime 97.4%). It does NOT drift toward the 31%
static fraction (which would mean workers turned into clean-decoders — the
forbidden divergence). The model-section "0.0%" line is a span-classification
artifact (`worker.decode` is unclassified in the model's stage map); the
load-bearing number is the causal §[1] 90.2%.

The causal §[5] "PRIMARY KEY-MISMATCH 97%" remediation is the #C item — it is a
FAITHFUL consequence of speculative pre-confirmation keying (see ledger #C), not a
pattern divergence to fix. The Fulcrum "remediation" text is a hypothesis
generator; the verdict is the ruling above.

---

## Residual = engine speed (the accepted SLOW)

Every Fulcrum-visible gap traces to decode-engine rate, not shape:

- Top Δbusy rows are all DECODE compute: `worker.block_body` +1872.6ms,
  `worker.decode` +1836.1ms, `worker.isal_stream_inflate` +1513.8ms.
- `fulcrum schedule` arbiter: **RATE-dominant** — consumer stalls 100% RATE
  (frontier not decoded), PLACEMENT 0.0%; "lever is decode speed (~15% bounded)".
- The publish-chain binds the wall, but the model shows the worker-bound knee
  CAPS L_resolve: at rapidgzip's faster L_resolve the publish-chain term (489.5ms)
  drops below gzippy's worker-bound term (499.1ms) — so resolve speed stops paying
  there; decode rate is the real binder.

This is the mandated "faithful but slow": gzippy's pure-Rust engine is ~1.85×
slower per decode than ISA-L, and that single substitution explains the wall gap.
The EARNED next step (engine optimization) is planned separately in
`plans/engine-optimization-plan.md` — NOT executed in this arc.
