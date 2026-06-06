# Plan: close structural gap to rapidgzip (parallel single-member)

**Branch:** `reimplement-isa-l` (`--no-default-features --features pure-rust-inflate`)  
**Vendor blueprint:** `vendor/rapidgzip/librapidarchive/src/rapidgzip/chunkdecoding/`  
**Bench:** `scripts/bench/run_locked_fulcrum.sh` (neurotic, frozen, interleaved N≥9, sha-verified)  
**Verdict instrument:** causal perturbation only — Fulcrum proposes, slow-inject/oracle removal disposes.

**Wall reference (pre-closure):** gzippy T16 ~1.55s vs rapidgzip ~0.47s; Fulcrum `worker.block_body` ~4790ms busy vs vendor 0ms; ~90% runtime window-absent.

**Control-plane status (June 2026 alignment):** `run_decode_task` uses exact `WindowMap::get(start_bit)` only; no worker publish; `try_speculative_decode_candidate` always window-absent; eager `queue_prefetched_marker_postprocess` uses full scan (`None` key).

---

## Done criterion

1. Structure mirrors vendor decode pipeline (dispatch tree, tryToDecode, window publish, post-process).
2. Wall TIE-or-better vs rapidgzip on silesia-large × T8/T16 (`scripts/measure.sh` / locked Fulcrum).
3. All 850+ lib tests + routing tests pass; output byte-identical.

---

## Closure phases (strict order — later phases assume earlier oracles)

### Phase 0 — Validate instruments (gate, ~1 session)

| Step | Action | Pass |
|------|--------|------|
| 0.1 | Re-bench post-alignment uncommitted changes | Baseline wall recorded in `wall-progress.md` |
| 0.2 | `fulcrum causal` on T8 trace | `worker.block_body` still wc ⇒ decode engine is lever |
| 0.3 | Oracle: single-chunk known window + known until | gzippy bytes == rapidgzip bytes |

### Phase 1 — P0: vendor decode dispatch tree (wall lever)

**Goal:** Replace MarkerRing-only path with vendor `decodeChunk` / `decodeChunkWithRapidgzip` / `finishDecodeChunkWithInexactOffset` / `decodeChunkWithInflateWrapper`.

| # | Function(s) | File | Vendor ref | Change |
|---|-------------|------|------------|--------|
| 1.1 | `finish_decode_chunk_with_inexact_offset` (**new**) | `gzip_chunk.rs` | `GzipChunk.hpp:280-410` | Port ISA-L loop: `setWindow`, `setStoppingPoints(END_OF_BLOCK \| END_OF_BLOCK_HEADER \| END_OF_STREAM_HEADER)`, preemptive stop at `untilOffset`, `append` buffers, `finalizeChunk` |
| 1.2 | `decode_chunk` dispatch | `gzip_chunk.rs` | `GzipChunk.hpp:661-710` | `if window && until_exact` → inflate wrapper; `else if window` → `decode_chunk_with_rapidgzip`; else window-absent unified path |
| 1.3 | `decode_chunk_with_rapidgzip` / `_impl` | `gzip_chunk.rs` | `hpp:413-654` | On `initialWindow` + ISAL: immediate `finish_decode...` (`hpp:440-444`); else `setInitialWindow` + block loop; at `clean_data_count >= 32768` → `finish_decode...` (`hpp:521-525`) |
| 1.4 | `decode_chunk_unified_marker` | `gzip_chunk.rs` | marker phase only | Shrink to window-absent bootstrap; phase 2 must call 1.1 not MarkerRing clean loop |
| 1.5 | `marker_decode_step` + `MarkerRing` | `gzip_chunk.rs`, `isal_lut_bulk.rs` | `deflate::Block` | Bootstrap-only: header 3-bit read, flip at vendor predicate, then hand off to 1.1 |
| 1.6 | `resumable_resync` | `gzip_chunk.rs` | (none on success) | Restrict to internal bad-seed recovery inside tryToDecode; remove from window-present path |
| 1.7 | `run_decode_task` | `chunk_fetcher.rs` | `GzipChunkFetcher.hpp:712-729` | When `get(hit)` and stop hint is chain-exact, pass `until_exact=true` into 1.2 |
| 1.8 | `IsalInflateWrapper` | `inflate_wrapper.rs` | `IsalInflateWrapper.hpp` | Ensure `stoppedAt`, `isFinalBlock`, `compressionType`, `tellCompressed` support preemptive stop bits |

**Bit contracts (must not drift):**

- Block header: 3 bits LSB-first — BFINAL(1) + BTYPE(2).
- Stored: align byte; LEN/NLEN 16+16 LE, `len == !nlen`; tryToDecode seeks to `offset.second` (see Phase 2).
- Flip: `distance_to_last_marker >= 65536` OR (`>= 32768` AND `== decoded_bytes`) — `deflate.hpp:1282-1284`.
- Marker: `u16 = 32768 + (32768 + pos + i - dist)`.
- Stop hint: inexact until — first block header with `tell() >= until` unless last/fixed exception (`hpp:550-555`, `hpp:339-344`).

**Oracle before merge:** single-member silesia chunk with oracle window → byte match libdeflate + rapidgzip; Fulcrum `worker.block_body` busy drops on window-present chunks.

### Phase 2 — P1: block finder + tryToDecode seek semantics

| # | Function(s) | File | Vendor ref | Change |
|---|-------------|------|------------|--------|
| 2.1 | `BlockBoundary` | `block_finder.rs` | pair in `tryToDecode` | Add `seek_bit: usize` (= `.second`; dynamic: `bit_offset`, stored: `byte_bit - 3`) |
| 2.2 | `find_uncompressed_blocks` | `block_finder.rs` | `Uncompressed.hpp:21-95` | Emit both `earliest` and `seek_bit` |
| 2.3 | `find_blocks` / search loop | `block_finder.rs`, `raw_block_finder.rs` | `hpp:803-845` | Replace merge-sort with inline alternating `findNextDynamic` / `findNextUncompressed` per 8 KiB chunk |
| 2.4 | `try_speculative_decode_candidate` | `chunk_fetcher.rs` | `hpp:712-734` | Accept `OffsetPair`; decode at `seek_bit`; metadata `encoded=first`, `max=seek` |
| 2.5 | `speculative_decode_find_boundary` | `chunk_fetcher.rs` | `hpp:796-846` | Wire 2.3 loop; pass pairs to 2.4 |

### Phase 3 — P1: window map cleanup (structural fidelity)

| # | Function(s) | File | Change |
|---|-------------|------|--------|
| 3.1 | `promote_speculative`, `evict_speculative`, `insert_speculative` | `window_map.rs` | Delete API + storage |
| 3.2 | `consumer_loop` accept/reject | `chunk_fetcher.rs` | Remove promote/evict calls; vendor discard-only on reject |
| 3.3 | `consumer_loop` accept guard | `chunk_fetcher.rs` | After chain proven: relax to `matches_encoded_offset(decode_start)` only (vendor `hpp:396-403`) |

### Phase 4 — P0: data plane (publish chain throughput)

| # | Function(s) | File | Change |
|---|-------------|------|--------|
| 4.1 | `ChunkData::get_last_window_vec` | `chunk_data.rs` | O(32K) tail extraction — contiguous ring or vendor-style single buffer |
| 4.2 | `SegmentedU8` tail | `rpmalloc_alloc/types.rs` | `copy_last_32k` without multi-segment walk on hot path |
| 4.3 | `prefill_window_prefix` / `data_prefix_len` | `chunk_data.rs` | Delete after Phase 1 inflate wrapper lands (vendor uses `setInitialWindow` only) |

### Phase 5 — P2: scheduling + docs + cleanup

| # | Action |
|---|--------|
| 5.1 | `submit_decode_to_pool`: priority 0 for all (perturb first) |
| 5.2 | `submit_post_process_to_pool`: priority −1 (vendor) |
| 5.3 | Update `docs/structural-gap-rapidgzip.md` — remove stale `get_at_worker` / handoff text |
| 5.4 | Gate `decode_bypass` / `sleep_decode` to env-only |
| 5.5 | Record survived-disproof findings in `wall-progress.md` only |

---

## Parallel implementation workstreams (subagents)

| Agent | Phase | Scope | Files (exclusive) |
|-------|-------|-------|-------------------|
| **A** | 1.1–1.6 | `finish_decode_chunk_with_inexact_offset` + gzip dispatch tree | `gzip_chunk.rs`, `inflate_wrapper.rs` (read-only extend if needed) |
| **B** | 1.7 | `run_decode_task` until-exact branch | `chunk_fetcher.rs` (`run_decode_task` + `DecodeParams` if needed) |
| **C** | 2.1–2.5 | Block finder pairs + alternating search | `block_finder.rs`, `raw_block_finder.rs`, `chunk_fetcher.rs` (`speculative_*` only) |
| **D** | 3.1–3.2 | Remove speculative WindowMap side-slot | `window_map.rs`, `chunk_fetcher.rs` (consumer promote/evict only) |

**Merge order:** A → B (B calls A's new API); C and D independent; resolve `chunk_fetcher.rs` conflicts by section.

---

## Verification checklist (every phase)

```bash
cargo test --release -q
cargo test --release routing -q
make   # local 30s guard
# After merge:
BRANCH=reimplement-isa-l THREADS=16 N=9 scripts/bench/run_locked_fulcrum.sh
```

Assert: CRC + ISIZE; `fulcrum stats` shows `worker.isal_stream_inflate` rising and `worker.block_body` falling on window-present fraction.

---

## Explicit non-goals (this plan)

- Novel inner-loop techniques (authorized separately in `CLAUDE.md` inner-loop scope).
- arm64 parallel SM (still gated off ISA-L).
- Thread priority revert without perturbation (Phase 5 only).

---

## References

- Prior matrix: [`sm-parity-gap-matrix.md`](sm-parity-gap-matrix.md)
- Survived findings: [`wall-progress.md`](wall-progress.md)
- Falsified levers: [`x86-falsification-ledger.md`](x86-falsification-ledger.md)
