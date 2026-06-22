# SCAFFOLD STRUCTURE MAP — gz T1 contig driver vs igzip monolith / rapidgzip chunk path

**Status: HYPOTHESIS-tier code-read (UNVALIDATED). NOT a finding.** Per the anti-bias
preamble (CLAUDE.md / MEMORY.md) a source-read is HYPOTHESIS, never a verdict — the
measurement leader's removal-oracle is the sole oracle. Every ranking, magnitude, and
per-block/per-byte attribution below is an UNVALIDATED candidate to be MEASURED, not a
conclusion. This doc exists only to PRIORITIZE which sub-regions the leader oracles first.

Author role: read-only structural-mapping helper (no benchmarks, no box access).
Branch: origin/kernel-converge-A @ 7c15266d. Read-only except this doc.

## Context recap (gated, from project_x86_decomposition_2026_06_21)

The cross-arch-consistent T1 lever vs igzip is the **CLEAN SCAFFOLD** = gz's contig
driver wrapping vs igzip's tight serial monolith: 6–16% on BOTH Intel and AMD,
CRC-excluded, dominant where the inner kernel is null (silesia, squishy). The
decomposition already LOCALIZED the aggregate to "gz contig driver vs igzip monolith"
but did NOT sub-localize WITHIN the driver. That sub-localization is what the leader
now oracles; this map ranks the candidates.

### Two cost layers (do not conflate them)

- **Layer B — MINIMAL contig driver** (what the decomposition's `cheap`/`thin` arms in
  `examples/streaming_thin.rs` actually measured = the 6–16%): a single sliding buffer
  `[retained 32768 | batch | SLOP]`, memmove-retain 32 KiB per flush, per-batch sink.
  This is ALREADY close to the monolith and STILL loses 6–16%.
- **Layer A — PRODUCTION T1 driver** (heavier; the thing the campaign must converge):
  `single_member::decompress_parallel(...,1)` → `sm_driver` routes T1 to
  `chunk_fetcher::drive_thin_t1_oracle` (DEFAULT at T1 — `use_thin_t1`,
  sm_driver.rs:190-217) → per-chunk `chunk_decode::decode_chunk` building a full
  `ChunkData`. Route-A oracle (streaming_thin `prod` − `gzippy`) bounded these
  production-only extras at ~5–10% of prod wall (project_route_a_bound_2026_06_21).

CRITICAL ROUTING FACT (Gate-4-style, verify before oracling): on the **gzippy-isal**
build the production T1 clean tail decodes through **REAL ISA-L `_04`** per chunk
(`finish_decode_chunk_isal_oracle`, chunk_decode.rs:376-658, gated by
`isal_engine_oracle_enabled()==cfg!(isal_clean_tail)`). So on gzippy-isal at T1 the
inner kernel is IDENTICAL to igzip — the entire gz-vs-igzip T1 delta is the DRIVER
wrapping ISA-L. That is exactly the "clean scaffold". (On gzippy-native the clean tail
is the pure-Rust `decode_clean_into_contig` kernel instead.)

### SOTA counterparts — nuance the leader must keep in view

- **igzip monolith** (`vendor/isa-l/igzip/igzip_inflate.c`, `isal_inflate` :2239-~2528,
  `read_header` :1399): ONE call over the whole stream. State inited ONCE. 32 KiB
  history kept implicitly in a `tmp_out` double-buffer (`2*ISAL_DEF_HIST_SIZE`, :2342).
  Decodes into `tmp_out` then `memcpy`s `tmp_out→user out` (:2436-2447). NO chunk
  lifecycle, NO block-boundary recording, NO subchunk split, NO per-chunk alloc, NO
  per-chunk window handoff. CRC folded inline (`update_checksum`, :2453). THIS is the
  "tight serial monolith" target shape.
- **rapidgzip chunk path** (`GzipChunk.hpp` `finishDecodeChunkWithInexactOffset`
  :282-410, `decodeChunkWithRapidgzip` :414-560): the faithful-port BLUEPRINT gz mirrors
  — but note rg's chunk path is ALSO NOT a monolith: it allocates `DecodedVector(128KiB)`
  per readStream iteration (:310), `result.append(...)` segmented (:379), records
  `appendDeflateBlockBoundary` per block-start (:364), splits subchunks, extracts a
  per-chunk window (`getLastWindow`, :523). So gz's per-block boundary/subchunk/segmented
  work is FAITHFUL TO rg yet DIVERGENT FROM the igzip monolith. **Tension to flag for the
  strategic fork:** "converge to the igzip monolith" and "faithfully port rg's chunk
  path" point in DIFFERENT directions for Layer-A bookkeeping. The leader's oracle on
  each sub-region tells us which sub-regions are pure cost (safe to shed toward monolith)
  vs load-bearing for the parallel pipeline (must keep for T>1).

---

## RANKED CANDIDATES (HYPOTHESIS — leader oracles in this order)

Ranking is by guessed scaffold-cost contribution. CONFIDENCE is in the EXISTENCE of the
structural delta (high — it is in the source); magnitude/critical-path is HYPOTHESIS.

### #1 — Per-chunk output data-plane: ChunkData alloc + ratio reserve, NO recycling pool at T1
- gz: `drive_thin_t1_oracle` calls `decode_chunk` (chunk_fetcher.rs:545) → `ChunkData::new`
  (chunk_decode.rs:1215; chunk_data.rs:306-353) per chunk, fresh `SegmentedU8` data +
  `SegmentedU16` markers. `finish_decode_chunk_isal_oracle` reserves
  `compute_initial_reserve(...)` ≥ 4 MiB / ≤ 64 MiB per chunk (chunk_decode.rs:351-374,
  461-476) and ISA-L grows into it. **`drive_thin_t1_oracle` has NO worker buffer-recycler**
  (it calls `ChunkData::new`, not `new_with_buffers`; chunk_data.rs:355-367) — so every
  chunk's multi-MiB buffer is a fresh alloc + first-touch page faults, then dropped.
- SOTA: igzip reuses ONE `tmp_out` (64 KiB) + the caller's out buffer for the whole stream
  (igzip_inflate.c:2342-2447) — ZERO per-chunk allocation. rg recycles via `BlockFetcher` +
  `FasterVector`/`RpmallocAllocator` (core/FasterVector.hpp:46-128; role map).
- Extra work gz does: per-chunk large reserve + first-touch faulting + drop, on the SAME
  thread serially. Ties to the DOCUMENTED page-fault gap (40% gz vs 17% rg; single_member.rs
  :236-243 "no pre-warm" note; chunk_data.rs:308-327 footprint-align reverted).
- Scaling: **per-chunk** alloc/fault (≈ ISIZE / 1 MiB chunks at T1 default) + per-byte
  segment commit. **One-time-ish** if a global pool happens to stay warm — leader confirm
  whether `chunk_buffer_pool` recycles across serial `decode_chunk` calls or not.
- Oracle hint: pin reserve to a recycled fixed buffer (the existing
  `GZIPPY_RESIDENT_OUTPUT_POOL` knob, chunk_decode.rs:362) and/or sweep `GZIPPY_CHUNK_KIB`;
  compare minflt + wall. HYPOTHESIS: largest single contributor (page-fault-bound).

### #2 — Per-chunk rolling-window handoff (alloc + 32 KiB memcpy + ISA-L re-seed)
- gz: `drive_thin_t1_oracle` rolls the window every chunk — `prev_tail = vec![0u8;32768]`
  + `copy_last_into` / `copy_range_into` (chunk_fetcher.rs:561-574), then passes it as
  `initial_window`; ISA-L is re-init + window-set per chunk inside the clean-tail path
  (`set_window`, finish path). A fresh 32 KiB alloc + memcpy PER CHUNK.
- SOTA: igzip keeps history implicitly in `tmp_out`; ZERO per-chunk window handoff
  (igzip_inflate.c, no analog). rg DOES extract a per-chunk window (`getLastWindow`,
  GzipChunk.hpp:523) so this is faithful-to-rg but divergent-from-monolith.
- Extra work gz does: window materialize + copy + decoder re-seed each chunk; ISA-L cold
  re-entry per chunk vs igzip's single warm pass.
- Scaling: **per-chunk**.
- Oracle hint: raise chunk size to 1 (single chunk) via `GZIPPY_CHUNK_KIB` huge value at
  T1 and compare to igzip; the window-roll + re-seed cost should collapse toward zero.

### #3 — Per-chunk ISA-L wrapper lifecycle (bounded re-slice, truncate/commit dance, boundary replay)
- gz: `finish_decode_chunk_isal_oracle` (chunk_decode.rs:376-658) per chunk: slices input to
  `stop_byte + 256 KiB` (:403-405), `decompress_deflate_from_bit_into_growable`,
  `truncate(decode_start)` then `commit(keep_len)` (:496,610), a boundary REPLAY loop
  (`note_inner_decoded_bytes` + `append_block_boundary_at` per recorded boundary, :636-653),
  then `finalize_with_deflate`. Each chunk re-enters ISA-L COLD.
- SOTA: igzip = one `isal_inflate` over the whole stream, no re-slice, no replay
  (igzip_inflate.c:2239+). rg's analog is `finishDecodeChunkWithInexactOffset` :282-410
  (faithful) but rg amortizes via the pool/threadpool.
- Extra work gz does: per-chunk FFI setup + the commit/replay accounting around ISA-L.
- Scaling: **per-chunk** (setup) + **per-block** (boundary replay).
- Oracle hint: oracle the boundary-replay loop (:636-653) separately from the FFI call.

### #4 — Per-block boundary recording + subchunk splitting (parallel-scaffold that survives at T1)
- gz: `append_block_boundary_at` (chunk_data.rs:944-1003) fires per EOB; `note_inner_decoded_bytes`
  feeds subchunk split sizing; called from the contig loop (chunk_decode.rs:1040, :648) and
  the isal replay. Pure parallel-pipeline metadata (block_map / prefetcher / block_finder
  index feed) that does NOTHING at T1 but is still computed.
- SOTA: igzip records NO boundaries (igzip_inflate.c — none). rg DOES
  (`appendDeflateBlockBoundary`, GzipChunk.hpp:364,561) — faithful-to-rg, divergent-from-monolith.
- Extra work gz does: per-EOB dedup + split-threshold check + Subchunk bookkeeping.
- Scaling: **per-block** (scales with deflate block count, not bytes).
- Oracle hint: stub `append_block_boundary_at`/`note_inner_decoded_bytes` to no-op at T1
  (byte-transparent for serial) and measure. HYPOTHESIS: small per-block, but block-dense
  corpora (silesia gzip-9 ~17 EOB/chunk, see chunk_data.rs:967-973) amplify it.

### #5 — CRC32 second-touch (per-byte re-read of decoded output)
- gz: CRC is a SEPARATE pass that RE-READS just-decoded bytes:
  `last_crc.update(chunk.data.decoded_range(...))` (chunk_decode.rs:614-620 isal path,
  :1717-1721 native contig path, :1120-1125 wrapper path). A per-byte read pass over output
  AFTER it is written.
- SOTA: igzip folds CRC INLINE during decode (`update_checksum(state, start_out, ...)`,
  igzip_inflate.c:2453) — no separate output re-read.
- IMPORTANT HONESTY CAVEAT: the decomposition's 6–16% scaffold is **CRC-EXCLUDED** (cheap/thin
  skip CRC; igzip computes it — so the scaffold number is already gz-FAVORABLE). This candidate
  is therefore NOT inside the 6–16%; it is ADDITIONAL production overhead. Oracle it
  SEPARATELY with the existing `GZIPPY_ORACLE_CRC_OFF=1` knob (sm_driver.rs:162,233) vs
  igzip-with-CRC to size the true production gap.
- Scaling: **per-byte**.

### #6 — Layer-B minimal-driver internals: sliding-buffer staging, memmove-retain, per-batch sink
- gz: `streaming_thin.rs` `gzippy_thin`/`igzip_bare_cheap` — `buf.copy_within(pos-WINDOW..pos,0)`
  memmove-retain 32 KiB every flush (streaming_thin.rs:114,301,461) + per-batch
  `sink(&buf[..])` → BufWriter `write_all` (a full per-byte copy of output). Decodes DIRECTLY
  into one big sliding buffer (no tmp_out double-buffer).
- SOTA: igzip decodes into `tmp_out` (64 KiB) then `memcpy tmp_out→out` per call
  (igzip_inflate.c:2436-2447). So igzip ALSO pays a per-byte staging copy; the gz delta here
  is the per-batch memmove-retain + the sliding-buffer vs ring-buffer geometry, NOT obviously
  more per-byte copying. **This is the residual that holds the measured 6–16% AFTER header
  re-entry was ruled out** — the leader should sub-oracle WITHIN it (memmove-retain vs sink
  vs decode-target geometry).
- Scaling: **per-batch** (memmove-retain) + **per-byte** (sink).
- NOTE already-bounded: per-block HEADER re-entry is EMPIRICALLY ≤0.5% on all 8 cells
  (HEADER-ARTIFACT, igzipbare−igzipbarecheap; project_x86_decomposition). Do NOT re-oracle it.

### #7 (LOW) — One-time / near-zero candidates (oracle last or skip)
- Input padding copy `input.to_vec() + [0u8;64]` in the bare arms (streaming_thin.rs:264,431):
  **one-time** per decode, negligible. Production path does not copy input (slices it).
- `trace_v2::SpanGuard::begin` per block/chunk (chunk_decode.rs:973,1631,1654): env-gated
  instrument, OnceLock-guarded, byte-transparent. HYPOTHESIS ~0; confirm it compiles to a
  cheap branch when `GZIPPY_TIMELINE` unset.
- gzip header parse `gzip_format::read_header` (gzip_format.rs:47): **one-time**, shared with
  igzip's `isal_read_gzip_header`. Not scaffold.

---

## Summary table (HYPOTHESIS ranks — leader's oracle is the verdict)

| # | sub-region | gz anchor | SOTA counterpart | scaling | in 6–16%? |
|---|---|---|---|---|---|
| 1 | per-chunk alloc + ratio reserve, no T1 recycler | chunk_decode.rs:351-374,461-476; chunk_data.rs:306-353 | igzip tmp_out reuse :2342; rg FasterVector pool | per-chunk + per-byte | partial |
| 2 | per-chunk window roll + ISA-L re-seed | chunk_fetcher.rs:561-574 | igzip implicit history; rg getLastWindow :523 | per-chunk | partial |
| 3 | per-chunk ISA-L lifecycle + boundary replay | chunk_decode.rs:376-658 | igzip one isal_inflate :2239 | per-chunk + per-block | partial |
| 4 | per-block boundary record + subchunk split | chunk_data.rs:944-1003; chunk_decode.rs:648,1040 | igzip none; rg appendDeflateBlockBoundary :364 | per-block | yes |
| 5 | CRC32 second-touch (output re-read) | chunk_decode.rs:614-620,1717-1721 | igzip inline update_checksum :2453 | per-byte | NO (CRC-excluded) |
| 6 | sliding-buffer staging + memmove-retain + sink | streaming_thin.rs:114,301,461 | igzip tmp_out memcpy :2436-2447 | per-batch + per-byte | YES (the residual) |
| 7 | input pad copy / trace guards / header parse | streaming_thin.rs:264,431; gzip_format.rs:47 | igzip inline | one-time | no |

## Caveats stamped on this map
- HYPOTHESIS-tier source-read. The removal-oracle (Gate-2) is the verdict, not this ranking.
- TIME/CODE-STAMPED: origin/kernel-converge-A @ 7c15266d. `git diff` before relying later.
- Magnitudes are guesses; the decomposition gives only the AGGREGATE (6–16%, CRC-excluded).
- Convergence-vs-blueprint TENSION (candidates #2/#3/#4): faithful to rg's chunk path but
  divergent from igzip's monolith — a strategic R3 call, not a pure cleanup. Flag to user.
