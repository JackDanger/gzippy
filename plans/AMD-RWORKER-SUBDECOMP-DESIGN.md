# R_WORKER sub-decomposition — partition DESIGN (for cursor-agent review)

## Context
Prior finding F-f81bd0c136c6 (fulcrum excess, Zen2, gated) localized the AMD gz-vs-rapidgzip
gap ENTIRELY to **R_WORKER** = the per-chunk decode call (huffman decode + commit + clean-CRC
+ ring), 94% of gz cycles, INSTRUCTION-bound (+256M instr on silesia, cache/branch-miss at
parity, instr ratio FLIPS 1.055 silesia / 0.865 nasa → silesia-specific ⇒ recoverable).
Now: split R_WORKER to find WHICH sub-step owns the +256M-instr silesia surplus.

R_WORKER currently wraps (gz) `decode_chunk_until_exact` and (rg) `GzipChunk.hpp` static
`decodeBlock`/`decodeChunkWithRapidgzip`, with ONE rdtsc pair per chunk, exclusive (depth
counter, OVERLAP_VIOLATIONS==0), /dev/null both arms.

## Matched structure (verified in source, both arms)
Both gz and rg drive a per-block loop with the SAME shape:

| step | gz (marker_inflate / chunk_decode) | rg (deflate.hpp / GzipChunk.hpp) |
|------|------------------------------------|----------------------------------|
| header / table build | `Block::read_header` (precode + code-lengths) **+ deferred** `build_huffman_luts_for_block` + `ensure_dist_hc` (built lazily on FIRST body `read`, lines 1854-1868) | `block->readHeader` → `readDynamicHuffmanCoding` (precode + code-lengths + LUT build, EAGER) |
| body decode | inner `while !eob { read_body → read_internal_compressed_specialized<MARKERS> }` (decode loops) + interleaved clean-CRC (`ContigFoldSink::update`) | inner `while !block->eob() { block->read(...); result.append(bufferViews) }` (decode + collect + clean-CRC inside append) |
| ring/commit + chunk overhead | ChunkData setup, window materialize, fold-drain, finalize, boundary append | ChunkData setup, `appendDeflateBlockBoundary`, footer, collect |

**ASYMMETRY CAUGHT (pre-review):** gz builds the decode LUTs (`build_huffman_luts_for_block`
+ `ensure_dist_hc`) LAZILY on the first body `read`, NOT in `read_header`. rg builds them
EAGERLY in `readHeader`. ⇒ to keep `table_build` matched, gz `table_build` MUST include that
LUT-build prologue (lines 1854-1868), else gz table_build is artificially cheap and the cost
leaks into gz body_decode → mislabel.

## Proposed sub-partition of R_WORKER (exclusive, sequential leaves, ONE rdtsc/block, symmetric)
1. **table_build** — readHeader (both) + (gz only) the deferred LUT prologue
   `build_huffman_luts_for_block`+`ensure_dist_hc`. Precode + code-lengths + litlen/dist LUTs.
2. **clean_decode** — body inner-loop of WINDOW-PRESENT chunks (decode + collect + interleaved
   clean-CRC).
3. **marker_decode** — body inner-loop of WINDOW-ABSENT (speculative) chunks. ACQUITTED; kept
   for conservation. Split clean-vs-marker at the CHUNK level by `initialWindow.is_some()` (gz
   `window present`) / rg `initialWindow` arg — a per-chunk bool, NOT per-block.
4. **clean_crc** — see OPEN QUESTION below.
5. **ring_other** — residual = R_WORKER − (table_build + clean_decode + marker_decode [+clean_crc]).
   Computed externally (chunk setup, boundary append, fold-drain, footer).

Exclusivity: table_build and body run sequentially within a block (never nested); body spans
of clean vs marker chunks are disjoint by chunk; residual is arithmetic. Sub-partition uses a
SEPARATE depth counter from the outer R_WORKER span (the sub-spans are nested INSIDE R_WORKER,
so they cannot share R_WORKER's depth counter — they'd all be overlap violations). The
sub-partition's own OVERLAP_VIOLATIONS must be 0 among the sub-spans themselves.

## Byte denominators (the denominator lesson — matched gz/rg, per sub-region)
- table_build bytes = total decoded output (identical gz/rg per corpus) → ratio = cyc_gz/cyc_rg.
- clean_decode bytes = decoded bytes produced by window-present chunks (identical gz/rg).
- marker_decode bytes = decoded bytes produced by window-absent chunks (identical gz/rg).
- ring_other bytes = total output.
All denominators are IDENTICAL across gz and rg for a given corpus (same chunking, same
output), so each region's gz/rg ratio is an honest absolute-cycle ratio. NOT a wall-total over
a subset (the Zen2 acquittal mismatch).

## OPEN QUESTIONS for cursor-agent
Q1. **clean_crc isolation.** Clean-CRC is INTERLEAVED inside the body inner-loop in BOTH arms
   (gz `ContigFoldSink::update` calls `crc.update` per emitted run; rg CRC is inside
   `result.append`). Isolating it as an exclusive leaf would need per-RUN rdtsc wrapping
   (high perturbation, violates the one-rdtsc-per-block discipline). Options:
   (a) FOLD clean_crc into clean_decode (document the deviation; the table_build-vs-body fork
       — the +256M-instr hypothesis's primary discriminator — is preserved).
   (b) DIFFERENTIAL: run a 2nd capture with CRC disabled in BOTH arms (gz has a no-crc switch
       `fold_nocrc_enabled`; rg needs `--no-crc`/equiv), clean_crc = body(crc) − body(nocrc).
   Which preserves Gate-0 / is least likely to mislabel?
Q2. Is the gz LUT-prologue-into-table_build mapping correct/sufficient for symmetry, or is
   any other table cost still deferred into the decode loop?
Q3. Are these sub-regions exclusive & non-overlapping (OVERLAP=0) given they nest inside the
   existing R_WORKER span? Is the separate-depth-counter approach right?
Q4. Are the per-sub-region byte denominators matched gz-vs-rg AND per-sub-region (not total)?
   Is using total-output for table_build/ring acceptable (since it's identical both arms, the
   ratio is honest), or must table_build use a per-block-count denominator?
Q5. Given the +256M-instr / +23%-branch silesia-specific surplus with cache at parity, WHICH
   sub-step would it most plausibly live in (table_build vs clean_decode), and would this
   partition ISOLATE it?
Q6. What would make `fulcrum excess` MIS-LABEL a sub-region here? (e.g. clean/marker bleed,
   LUT-deferral leak, denominator mismatch, residual absorbing a real cost.)
Q7. instr metric: per-region instructions are NOT measurable via rdtsc spans (cycles only).
   Per-region rdpmc in BOTH a Rust and a C++ codebase across 4 worker threads is high-risk
   (per-thread perf_event_open+mmap, easy to make inert → Gate-0 fail). On a FROZEN box
   (boost=0, fixed freq) with IPC-parity ESTABLISHED by the prior finding (branch-miss + cache
   at parity), is per-sub-region cyc/byte excess an acceptable instruction-surplus localizer,
   with per-region instr flagged as OWED? Or is rdpmc mandatory?
