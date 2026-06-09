# PIPELINE-FIDELITY AUDIT — verdict (2026-06-09, lead-auditor turn)

Branch reimplement-isa-l, HEAD d56cb0f5 (read-only on src/decompress/parallel/; no
implementation this turn). Method: first-hand SOURCE comparison of gzippy's
coordination + serial tail against the ACTUAL rapidgzip headers
(`vendor/rapidgzip/librapidarchive/src/{rapidgzip,core}/`). No perturbation run
(box held by a residual-sizing oracle); rankings are reasoned-magnitude, NOT
measured — flagged inline. Agent/subagent tool ABSENT (verified) ⇒ sequential,
self-disproof only. OWES Supervisor Opus gate; NOT advisor-vetted.

---

## HEADLINE

The thread-coordination and serial-processing STRUCTURE is **largely faithful** to
rapidgzip. I did **not** find a single large coordination/serial deviation that
accounts for the ~10% low-T residual (isal T1 0.899x / T4 0.900x). The prime
suspect ("we introduced a coordination/serial deviation") is **only weakly borne
out**: the deviations that exist are real but individually small and diffuse
(per-chunk throwaway allocations, unconditional per-iteration scans, a one-time
chunk-0 full-window, an output over-reserve). The residual is more consistent with
a **diffuse per-chunk constant-factor** than with one structural divergence.

The single most load-bearing correction for the campaign:

> **The "per-chunk FFI handoff" residual term is NOT a gzippy deviation — it is
> MATCHED to rapidgzip.** gzippy's per-chunk ISA-L setup (zero an `inflate_state`,
> `isal_inflate_init`, `isal_inflate_set_dict` of the 32 KiB window, per-block
> boundary `Vec`) is exactly what rapidgzip's `IsalInflateWrapper` does in its
> WITH_ISAL build (which STATE.md confirms is the comparison baseline: "rg also
> uses ISA-L at T1"). So this term cannot explain the gap vs rg; both pay it.

This means the low-T deficit is mostly NOT in the coordination/serial code as
hypothesized — it is most likely in the **per-chunk inner ISA-L call constant
factor itself** (which both pay, but gzippy may pay a heavier instance of — see D4)
plus a thin sum of the small deviations below.

---

## RANKED DEVIATIONS (gzippy emits work rapidgzip does not / differs structurally)

Rank = likely low-T wall impact (best estimate). Confidence flagged. Each:
vendor file:line ↔ gzippy file:line · runtime work · convergence target.

### D1 — Per-chunk ISA-L output OVER-RESERVE vs vendor incremental growth  [Tier-2, conf: med, footprint-dominant]
- gzippy `gzip_chunk.rs:263-274` (`reserve_len = compressed_span × 8`, floor 4 MiB,
  cap 64 MiB) → `segmented_buffer.rs:243 writable_tail_reserve(min_spare)` →
  `ensure_buf` → `chunk_buffer_pool::take_u8(min_capacity)` reserves ONE contiguous
  region sized to ~8× the compressed span per chunk so ISA-L never stops for output
  (it returns `None` on under-reserve rather than realloc).
- vendor `IsalInflateWrapper`/`GzipChunk` decode into the `ChunkData` FasterVector
  and grow it **incrementally** as ISA-L produces output (loop with bounded
  `avail_out`, extend on demand).
- Runtime work: a larger-than-needed contiguous allocator request per chunk → bigger
  per-worker resident working set + buffer-pool churn. This is the structural source
  of the long-known "40% page-fault vs vendor 17%" footprint gap. `poison_reserved_tail`
  is a **no-op in release** (`segmented_buffer.rs:59`, only `#[cfg(test)]+env`), so
  the over-reserve does NOT pre-fault the tail — the cost is allocation size / cache
  residency, NOT a zeroing pass.
- Magnitude: high-T (memory-bandwidth) more than T1 (single reused buffer). Honest
  caveat: at T1 the pool reuses one buffer, so this is a weaker T1 lever than its
  rank suggests; it is ranked #1 because it is the clearest *named-structural*
  divergence in the serial decode path.
- Converge: grow the chunk output incrementally (loop ISA-L with a bounded
  `avail_out` window like vendor) instead of one 8× contiguous reserve; or size the
  reserve to the realized decoded ratio rather than 8×.

### D2 — Unconditional per-iteration prefetch-promote + harvest at consumer top  [Tier-1, conf: high (structural), T-scaling]
- gzippy `chunk_fetcher.rs:1213 process_ready_prefetches()` + `:1218
  harvest_ready_postprocess()` run at the TOP of EVERY consumer iteration.
- vendor read loop `ParallelGzipReader.hpp:575-643` has **neither**: prefetch
  promotion is internal to `BlockFetcher::get`, and marker-harvest happens ONLY
  inside `waitForReplacedMarkers` (`GzipChunkFetcher.hpp:497-511`) — i.e. only for
  marker chunks, only while blocking.
- Runtime work: a prefetch-map mutex lock + map scan, and a postprocess-map scan,
  per consumed chunk regardless of need.
- Magnitude: at T1 the maps are EMPTY (one lock + empty scan/chunk) — small; scales
  with T. Real structural divergence, modest cost.
- Converge: fold promotion into the fetch (vendor shape) and harvest only inside the
  marker-wait, removing the unconditional top-of-loop scans.

### D3 — `queue_prefetched_marker_postprocess` full sorted cache scan on the CLEAN branch  [Tier-1, conf: high (structural)]
- gzippy `chunk_fetcher.rs:1715` (clean branch) + `:1825` (marker branch) both call
  the full sorted prefetch-cache scan.
- vendor calls `queuePrefetchedChunkPostProcessing` (`GzipChunkFetcher.hpp:520-551`)
  **only** from `waitForReplacedMarkers:513` — i.e. only when post-processing a
  MARKER chunk. A clean chunk in vendor never triggers this scan.
- Runtime work: a `prefetch_cache_contents_sorted()` Vec build + sort + per-entry
  `window_map.get` on every clean chunk.
- Magnitude: at T1 the prefetch cache is empty (cheap); grows with T. Structural
  divergence from vendor's "marker-chunks-only" trigger.
- Converge: gate the scan to the marker branch (vendor shape).

### D4 — Per-chunk throwaway `format!` trace Strings evaluated with tracing OFF  [Tier-2, conf: high, magnitude small]
- gzippy: `SpanGuard::begin_with(name, &format!(...))` and
  `emit_instant(name, &format!(...))` build the `format!` String in the CALLER
  before the `is_enabled()` gate inside the callee — so production (trace OFF)
  still allocates+frees the String. Per clean chunk on the serial consumer this hits
  at least `chunk_fetcher.rs:1690` (window_publish_clean), `:1697`
  (causal.window_publish), and `:1578` (block_fetcher_get on a cold get); marker
  chunks add `:1788`, `:1814`, `:1872`. (`SpanGuard`/`emit_instant` themselves ARE
  gated — `trace_v2.rs:231,247,194` — the leak is the eager arg.)
- vendor: no equivalent; its `now()`/stats are gated behind `m_statisticsEnabled`.
- Runtime work: ~2-3 heap String alloc+free per chunk on the in-order consumer.
- Magnitude: small (~µs-scale total) but it is a textbook "compiled code emits work
  vendor doesn't," and the cheapest convergence on the list.
- Converge: pass a closure/`format_args!` deferred behind `is_enabled()`, or guard
  each call site with `if trace_v2::is_enabled()`.

### D5 — Chunk-0 initial window: full 32 KiB ZEROED window vs vendor EMPTY window  [Tier-3, conf: high, one-time]
- gzippy `chunk_fetcher.rs:560-561`: `let zero_window = [0u8;32768];
  window_map.insert_bytes(0, &zero_window)` — chunk 0 gets a FULL 32 KiB window, so
  `finish_decode_chunk_isal_oracle` (`gzip_chunk.rs:223`) treats it as window-present
  and runs `isal_inflate_set_dict` with a 32 KiB **zero** dict.
- vendor `GzipChunkFetcher.hpp:100-106`: `m_windowMap->emplace(*firstBlockInStream,
  {}, CompressionType::NONE)` — an EMPTY window; ISA-L for chunk 0 runs with no
  set_dict.
- Runtime work: a 32 KiB stack/heap zero-fill + a 32 KiB `isal_inflate_set_dict`
  memcpy that vendor skips, plus chunk 0 routed through the window-present path.
- Magnitude: one-time (chunk 0 only) — cannot explain a sustained deficit; it is the
  concrete mechanism behind the named "chunk-0 bootstrap" residual term, and it is
  small.
- Converge: insert an empty window for the first block and let chunk-0 decode with an
  empty dict (vendor shape).

### D6 — Unconditional `Instant::now()`/`.elapsed()` timing on the hot consumer path  [Tier-2, conf: high, magnitude negligible]
- gzippy `chunk_fetcher.rs:1207,1210,1221,1577` + `drain_one_pending`
  `:3814,3815,3856,3884,3977`: ~9-18 clock reads per chunk, NOT gated.
- vendor gates every `now()` behind `m_statisticsEnabled`
  (`ParallelGzipReader.hpp:613-616, 620-624`).
- Runtime work: ~9-18 vDSO `clock_gettime` (~25 ns each) per chunk.
- Magnitude: negligible (~µs total). Cleanup / faithfulness only.
- Converge: gate behind the trace/stats flag.

### D7 — `recycle_deferral` 2-deep buffer hold vs vendor immediate recycle  [Tier-4, conf: high, fragility signal]
- gzippy `chunk_fetcher.rs:1187 RECYCLE_DEFER_DEPTH=2` + `defer_chunk_recycle`:
  drained chunks' decode buffers are held 2 deep before returning to the pool. The
  comment attributes this to a "lone-drain CRC bisect 2026-06-05" — a correctness
  workaround, not a vendor behavior.
- vendor recycles a chunk's buffers immediately when `writeAll` completes
  (FasterVector auto-recycle; the writeFunctor at `ParallelGzipReader.hpp:521` owns
  the chunk only for the write).
- Runtime work: 2 extra chunks' buffers resident at all times → larger working set.
- Magnitude: footprint at high T; the deeper concern is that it signals an ORDERING
  FRAGILITY (a CRC race) absent in vendor — worth a faithful-port pass to remove the
  root cause rather than carry the deferral.
- Converge: find/remove the CRC ordering race so immediate recycle is safe (vendor
  shape).

---

## NEGATIVE FINDINGS — verified FAITHFUL (ruled out as deviations; high value)

These were checked first-hand and MATCH vendor; do not spend further effort here.

- **In-order OUTPUT write**: gzippy gathers zero-copy iovecs (`append_output_iovecs`
  → `writev`, `chunk_fetcher.rs:3896-3931`) == vendor `writeAll`/`toIoVec`
  (`ParallelGzipReader.hpp:521`). No extra copy. FAITHFUL.
- **OverlapWriter is OFF in production** (`output_writer.rs:49-52`, gated on
  `GZIPPY_OVERLAP_WRITER`, unset in prod). Production uses inline writev ==
  vendor's inline `writeFunctor`. DIS-5/DIS-11 confirmed; re-verified. FAITHFUL.
- **CRC32**: constant-time per-stream combine on the consumer (`total_crc.append`,
  `chunk_fetcher.rs:3899-3902`) == vendor `processCRC32` combine
  (`ParallelGzipReader.hpp:1454-1503`, `m_crc32.append` + per-footer combine). NO
  re-scan of decoded bytes on either side. FAITHFUL (gzippy's claim verified).
- **WindowMap.get**: zero-alloc `Arc<CompressedVector>` refcount bump on hit
  (`window_map.rs:76-91`) == vendor `shared_ptr<const Window>`. The stale comment at
  `chunk_fetcher.rs:1748` ("still allocates Arc<[u8;32768]> on hit") is WRONG —
  `confirmed_predecessor_window` does not allocate. FAITHFUL.
- **Prefetch sizing/admission**: `cache = max(16, pool)`, `prefetch = 2×pool`,
  `FetchMultiStream(3, 16)` (`chunk_fetcher.rs:175-177, 578-579`) — explicitly
  transliterated to vendor `BlockFetcher.hpp:181-185`. Prior divergent
  `BURST_PREFETCH`/`pool×2` knobs deleted. FAITHFUL.
- **ThreadPool at T1**: `thread_count==0` → run INLINE via deferred future
  (`thread_pool.rs:309-316`) == vendor `std::async(std::launch::deferred)` at
  `parallelization==1` (`m_threadPool(==1?0:...)`, `BlockFetcher.hpp:185`). No worker
  wakeup at T1. FAITHFUL. (Both allocate a per-task future/channel — vendor
  std::async shared-state vs gzippy `mpsc::sync_channel(1)` — roughly matched.)
- **Per-chunk ISA-L FFI handoff** (state zero + init + set_dict + boundary Vec,
  `isal_decompress.rs:810-923`): vendor's `IsalInflateWrapper` (WITH_ISAL) does the
  same per chunk. MATCHED — see Headline; this rules the named "FFI-handoff" residual
  term OUT as a deviation.
- **Window-publish handoff**: gzippy publishes the end-window on the consumer thread
  BEFORE post-process (`publish_end_window_before_post_process`,
  `chunk_fetcher.rs:1696,1812,2763`) == vendor `queueChunkForPostProcessing`
  emplacing `getLastWindow(*previousWindow)` on the main thread
  (`GzipChunkFetcher.hpp:557-575`), with the empty-footer-window special case
  matched (`:562-570` ↔ `chunk_end_uses_empty_footer_window`). FAITHFUL.

---

## RECOMMENDATION TO SUPERVISOR

1. The hunch is mostly NEGATED at the structural level. Re-point the low-T
   investigation away from "a coordination/serial structural deviation" toward the
   **per-chunk ISA-L inner-call constant factor** (D1 + the matched-but-heavy
   inner setup) and the **diffuse sum** of D2-D6.
2. The cheap, faithful, byte-transparent convergences (D2, D3, D4, D5, D6) are worth
   landing as a SINGLE bundle and measuring as one perturbation (per-discipline:
   interleaved, sha-verified, isal_chunks>=14 gated) — none individually justify a
   turn, but together they remove the diffuse constant factor the residual most
   resembles. D1 and D7 are larger faithful-port jobs (incremental output growth;
   removing the CRC-ordering race) — their own turns.
3. Owed: a real Opus disproof gate + a measured perturbation of the D2-D6 bundle.
   This turn is source-only by discipline.
