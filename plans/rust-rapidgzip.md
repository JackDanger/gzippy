# gzippy parallel-single-member: state, wisdom, next moves

This is the canonical reference for the parallel-single-member decode
pipeline (`src/decompress/parallel/`). It replaces the older
`plan.md` + every other file that used to live in `plans/`. Read it
end-to-end before changing parallel-SM code. Update it when something
in here goes stale.

The contents are:

1. **Where we are.** Current throughput, what's on the hot path, what's
   structurally locked.
2. **What the system is.** The pipeline anatomy from input slice to
   output writer, with file:fn citations.
3. **What we now know about doing this work.** Hard-won wisdom about
   measurement, advisors, reverts, and the patterns that produce real
   wins vs apparent ones.
4. **Opportunities for next.** Concrete next moves, ranked by the
   evidence we have for them, with the structural alternatives flagged
   when the surgical search is exhausted.
5. **Done-when criteria.** Four measurable gates that close the project.

The prime directive in `CLAUDE.md` still rules: **gzippy aims to be the
fastest gzip implementation ever created.** Everything below is in
service of that.

---

## 1. Where we are (as of 2026-05-26)

### Numbers

`make bench-sm` on neurotic, T=16, fixture = `silesia-large.gz`
(503 MB raw → 162 MB compressed, ~3 × silesia.tar concatenated):

| Metric | Value |
|---|---:|
| gzippy mean throughput | **772 MB/s** (was 740 before `8ef7a4e`) |
| gzippy best single run | **800 MB/s** |
| gzippy best raw wall | **0.630 s** |
| rapidgzip mean (same run) | 1251 MB/s |
| gzippy / rapidgzip ratio | **0.62×** (was 0.53×) |
| Total cycles per 5-iter perf record | ~62.9 B (pre-CRC-move) |

Compared to the session-start baseline at commit `73c4a21` (BitReader
8-byte refill), throughput is **+82 %** (423 → 772 MB/s) and the ratio
vs rapidgzip closed from 0.30× to 0.62×. The +17 % ratio jump from
0.53× → 0.62× came from commit `8ef7a4e` (move narrowed-bytes CRC32
from consumer to post-process worker, vendor parity).

### Note on absolute numbers vs ratio

The 1400 MB/s rapidgzip number that previously appeared in this doc
came from a less-loaded snapshot of neurotic; on the 2026-05-26 bench
both gzippy and rapidgzip wall numbers were lower (system busier).
The **ratio** is the controlled measurement — bench-sm runs both
binaries on the same host in the same window. Quote ratios when
comparing across sessions; absolute MB/s only inside a single run.

### Top of profile, post-iteration

`perf record -F 999 --call-graph=dwarf` on the latest commit:

| Symbol | % cycles | Notes |
|---|---:|---|
| `bootstrap_with_deflate_block` | **22.8 %** | Phase 1 marker-mode Huffman decode. Vendor-parity tight. |
| `__memmove_avx_unaligned_erms` (libc) | **21.9 %** | 47 % kernel page-fault path, 25 % back-ref copy, 28 % mixed. |
| `clear_page_erms` (kernel) | **17.8 %** | First-touch faults on chunk buffers. |
| `submit_post_process_to_pool` body | 6.1 % | `replace_markers_lut_narrow` LUT pass. |
| `..@37.end` / `..@42.end` / `loop_block` / `decode_len_dist` | ~13 % | ISA-L C library internals. |
| `__rmqueue_pcplist` (kernel) | 1.2 % | Page allocator. |

ISA-L internals (~13 %) are in `vendor/isa-l` and cannot be improved
without forking. Kernel paths (~20 % combined) are inherent to the
workload's memory pattern. **Roughly 40 % of cycles are off-limits to
surgical Rust changes.**

### What's structurally locked

- **Per-byte rate of bootstrap.** ~149 MB/s. Vendor's equivalent runs
  at ~250 MB/s. The ~1.25× per-byte Rust-vs-C++ codegen gap on the
  same algorithm is measured (`plans/`-era apples-to-apples bench).
  Closing it without leaving pure-Rust is not on the table.
- **First-touch page faults on chunk buffers.** Each new pool buffer
  costs ~4 KiB of kernel page-zero per page written. Pre-allocating
  capacity does NOT help (`0da3530` revert). The only fix is
  `MADV_POPULATE_WRITE` or reusing pre-faulted memory across
  invocations (structural).
- **Wall is bounded by MAX worker chunk time.** Wall ≈ 0.55-0.70 s,
  CPU usage ≈ 32 % across 16 logical cores. Workers are NOT
  CPU-bound — they're waiting on the slowest critical-path chunk.
  Cycle-reduction wins translate to wall reduction only if they hit
  the chunk that bounds the FIFO drain.

---

## 2. What the system is

The CLI calls `decompress::decompress_gzip_libdeflate` which routes
per the table in `CLAUDE.md` "Production Routing". Single-member
inputs ≥ 10 MiB on a multi-thread host with ISA-L available take
**`parallel::single_member::decompress_parallel`** — this section
covers that path.

### The pipeline, end to end

```
Input slice (mmap'd or read into Vec)
  │
  ▼
sm_driver.rs::read_parallel_sm  ──── trims gzip header + trailer,
  │                                  reads ISIZE / CRC32 from footer,
  │                                  builds ChunkConfiguration,
  │                                  calls chunk_fetcher::drive
  ▼
chunk_fetcher.rs::drive  ──── owns the ThreadPool (vendor JoiningThread
  │                            equivalent), the WindowMap, the BlockMap,
  │                            and the FIFO drain loop. One main thread.
  │
  ├─► [prefetcher] dispatches speculative decode tasks to workers,
  │   keyed by partition seed (a guess at where the next deflate
  │   block starts).
  │
  ├─► [workers]    ┌── try_speculative_decode_candidate(start_bit)
  │   (Tmax of      │     ├─ Phase 1 bootstrap: marker-emitting
  │   them)         │     │     deflate decode via Block::read +
  │                 │     │     IsalLitLenCode::decode + IsalDistCode +
  │                 │     │     emit_backref_ring. Output goes to
  │                 │     │     chunk.data_with_markers (u16). When
  │                 │     │     32 KiB of clean output materializes,
  │                 │     │     contains_marker_bytes flips → switch
  │                 │     │     to Phase 2.
  │                 │     ├─ Phase 2 ISA-L: hands off to
  │                 │     │     IsalInflateWrapper. ISA-L writes
  │                 │     │     DIRECTLY into chunk.data spare
  │                 │     │     capacity (no intermediate buffer).
  │                 │     │     The worker calls
  │                 │     │     note_inner_decoded_bytes per ISA-L
  │                 │     │     call and updates chunk.crc32s.last
  │                 │     │     inline after each outer iter.
  │                 │     └─ if speculation fails at start_bit:
  │                 │           with_sync_boundary_search scans the
  │                 │           input in CHUNK_SIZE_BITS windows on
  │                 │           this same worker thread (NO thread
  │                 │           spawn) and tries each candidate.
  │                 │
  │                 └── on success, returns ChunkData to the
  │                     BlockFetcher cache keyed by partition seed.
  │                     If the actual decode start ≠ seed, the
  │                     chunk is re-keyed under the real offset.
  │
  ├─► [consumer / main thread]
  │     ├─ rx.recv()s the next chunk from the FIFO.
  │     ├─ set_encoded_offset to match the prior chunk's actual end.
  │     ├─ Publishes the chunk's tail 32 KiB to the WindowMap.
  │     ├─ Submits a post_process task to the pool:
  │     │   ── if data_with_markers ≥ 128 Ki elements:
  │     │      replace_markers_lut_narrow fuses replace + narrow
  │     │      into a single u16→u8 pass (vendor
  │     │      DecodedData.hpp:316-337 pattern).
  │     │   ── otherwise: apply_window in place then narrow.
  │     │   ── populate_subchunk_windows uses the resulting
  │     │      narrowed buffer (no internal re-narrow).
  │     ├─ Awaits the post_process future.
  │     ├─ Drains: writes chunk.narrowed + chunk.data to the
  │     │   output writer. Combines worker CRCs (chunk.crc32s)
  │     │   into total_crc via the constant-time polynomial
  │     │   `append` (NO bytewise re-CRC of chunk.data).
  │     ├─ append_subchunks_to_block_map for successor routing.
  │     └─ Returns chunk to pool on Drop.
  │
  └─► Final: total_crc.verify(footer.crc32) + total_size ==
      footer.uncompressed_size. Either succeeds or returns a
      hard `Err(GzippyError::Decompression(_))` — NO fallback
      (CLAUDE.md Rule 5).
```

### Files that matter

| File | Role |
|---|---|
| `src/decompress/parallel/sm_driver.rs` | Entry; trims gzip header/trailer; reads expected CRC + ISIZE; calls `drive` and verifies the result. |
| `src/decompress/parallel/single_member.rs` | Public-facing routing wrapper; converts errors; tracks `MARKER_PIPELINE_RUNS` counter for the routing-trap test. |
| `src/decompress/parallel/chunk_fetcher.rs` | The main thread + worker pool + consumer drain. `drive`, `consumer_loop`, `drain_one_pending`, `run_decode_task`, `run_post_process_task`, `submit_post_process_to_pool`, `speculative_decode_find_boundary`, `try_speculative_decode_candidate`. |
| `src/decompress/parallel/gzip_chunk.rs` | The per-worker chunk decoder. `bootstrap_with_deflate_block`, `decode_chunk_isal_impl` (Phase 2 inline-into-chunk.data writer), `absorb_isal_tail`, `decode_chunk_marker_bootstrap_then_isal`. |
| `src/decompress/parallel/deflate_block.rs` | The pure-Rust deflate decoder for Phase 1. `Block::read`, `Block::read_internal_compressed_specialized<CONTAINS_MARKERS>`, `emit_backref_ring`. Inner Huffman loop lives here. |
| `src/decompress/parallel/isal_huffman.rs` | `IsalLitLenCode`, `IsalDistCode` — the multi-symbol LUT Huffman decoders. Both `#[inline(always)]`. |
| `src/decompress/inflate/consume_first_decode.rs` | `Bits` (libdeflate-shaped bit reader, branchless 8-byte refill). Shared with the sequential / BGZF decoders. |
| `src/decompress/parallel/chunk_data.rs` | The `ChunkData` struct: `data`, `data_with_markers`, `narrowed`, `crc32s`, `subchunks`, `footers`. Plus `append_clean`, `append_markered`, `note_inner_decoded_bytes`, `note_clean_bytes_written_in_place`, `populate_subchunk_windows`. |
| `src/decompress/parallel/replace_markers.rs` | `replace_markers_lut_narrow` (fused u16→u8 LUT pass for ≥128 Ki chunks); `replace_markers` (AVX2/scalar for smaller chunks). |
| `src/decompress/parallel/block_finder.rs` | The dynamic-Huffman block-boundary scanner with the 15-bit LUT + the 8-byte unaligned BitReader refill (`73c4a21`). |
| `src/decompress/parallel/raw_block_finder.rs` | `RawBlockFinderCoordinator` + `with_sync_boundary_search` (the sync slow-path replacement for the prior thread-spawning `with_scoped_boundary_search`). |
| `src/decompress/parallel/window_map.rs` | Inter-chunk window publication (consumer publishes the tail 32 KiB; successor worker reads it). |
| `src/decompress/parallel/block_map.rs` | Post-decode subchunk lookup index. |
| `src/decompress/parallel/chunk_buffer_pool.rs` | Per-worker recycling pool for `Vec<u8>` and `Vec<u16>` chunk buffers. Hits/misses tracked. |

### Routing facts that surprise people

- **Speculative chunks dominate the wall.** ~80 % of bench-sm chunks
  fail their first-try speculation at the prefetcher's partition seed
  and need a boundary search. Of those, EVERY attempt succeeds on the
  first BlockFinder candidate (per `Slow-path decode: ok=61 fail=0`
  in the verbose stats). The slow-path is invoked often but never
  has to walk a candidate list.
- **`contains_marker_bytes` rarely flips on speculative chunks.** For
  highly compressible inputs (silesia), back-refs span the entire
  chunk → no 32 KiB clean run materializes → the worker runs Phase 1
  bootstrap for the WHOLE chunk. Bench-sm `body_rate_MB/s ≈ 149`
  reports the bootstrap rate.
- **The wall is consumer-bound at first glance and worker-bound on
  closer reading.** The consumer thread is no longer the bottleneck
  (since the CRC fix at `e1beab4` and the `extend_from_slice` →
  `copy_nonoverlapping` fix at `630d44d`). Wall ≈ max(worker chunk
  time) per the strict-FIFO drain. The slowest single chunk's
  bootstrap time defines wall.

---

## 3. What we now know about doing this work

These are wisdom-class rules, in priority order. Each was learned the
expensive way.

### Measure before AND after every change. Two priors will not save you.

The single biggest historical failure mode of this project is
acting on stale profile data. Examples from THIS session alone:

- **The "floor" doc was wrong by 75 % throughput.** A previous session
  declared parallel-SM at its "architectural floor" at 356 ms / 423
  MB/s. The next session opened with no new measurement, applied
  several optimizations, and discovered the floor was actually
  closer to 740 MB/s — the previous "floor" was the profile state
  before three obvious wins.
- **Bootstrap-bench cycle shares don't apply to silesia.** A SIMD
  marker-scan plan was built on a bench-fixture profile showing
  ~7 % cycles in marker scanning. On the real silesia full-pipeline
  it was ≤ 1.5 %. The plan was abandoned mid-implementation after
  re-measurement.
- **Three reverts in a row** in this session, each one looked like
  an obvious win in source: pre-allocate `data_with_markers`,
  guard the hoisted refill, branch instead of the unified LUT.
  All three regressed cycles. CPU branch predictors and
  instruction-level parallelism are not what a Rust reader
  intuitively expects.

The discipline: every commit gets `make bench-sm` numbers and a fresh
`perf record` on neurotic. If both move the right way, ship. If either
regresses, revert immediately — there's nothing to learn from a
broken commit sitting on the branch.

### Use Opus advisors for sanity-checks, not for synthesis

Two patterns work for advisors:

- **Pre-flight cross-check.** "Here's my plan, here are the numbers
  I have, here are the unknowns. Find gaps." → useful. The advisor
  has no context coupling, sees the analysis fresh, and catches
  blind spots (wrong cycle-share extrapolation; assumed but
  unconfigured build flags; missed code paths).
- **Post-mortem audit.** "Here's the code I shipped, here's what
  failed, here's the diff. Find what I missed." → useful. The
  advisor walks the diff with adversarial eyes.

What does NOT work:

- **"Implement X based on your findings."** The advisor pushes
  synthesis off your context; the result is shallow and you have to
  re-derive everything to verify it.
- **Loops of advisors talking to each other.** Each round shifts
  the analysis to whichever frame the most recent advisor brought.
  Use ONE advisor pass per concrete question.

**Worktree gotcha (2026-05-26).** When the parallel-SM work runs inside a
git worktree (e.g. `/Users/jackdanger/www/gzippy/.git/worktrees/perf/...`),
the `Agent` tool spawns the subagent rooted at the **main repo dir**, not at
the spawning agent's Bash CWD. Putting `Working dir: <worktree>` in the prompt
TEXT is ignored. Symptom: the advisor reads files from `main`, doesn't see
your branch's commits, and disagrees with your premise on facts you have
verified locally. Confirmed when an advisor reading `Cargo.toml` reported no
`arena-allocator` feature while it was plainly at `Cargo.toml:39` on this
branch.

**Mitigation.** Step 0 of every advisor prompt must be an explicit
`cd <absolute-worktree-path> && pwd` with the instruction that every
subsequent Read/Grep MUST resolve relative to that directory, and to halt
with a clear error if `pwd` shows anything else. Without this, the round
trip is wasted.

### Neurotic measurements are noisy; characterize variance, don't chase single-run jumps

`bench-sm` runs on the homelab Intel i9. Wall variance is ~5-15 % CV
across runs (system load matters). A 10 % delta on a single run is
within noise. The patterns that ARE signal:

- **5+ run mean, both gzippy and rapidgzip.** rapidgzip's number is
  the "system speed" baseline; if both gzippy and rapidgzip moved
  together, it's neurotic load, not our code.
- **best-of-N min wall.** Sometimes more stable than mean — the
  best-case is bounded by the actual decode work, the rest is
  scheduling jitter.
- **Total cycles from `perf record`.** Much less noisy than wall.
  Cycle reduction is the right signal for "did my code do less
  work"; wall reduction needs the cycle reduction PLUS the savings
  landing on the critical path.

When wall doesn't move but cycles drop, the savings landed on
non-critical-path work (parallel headroom). When wall drops but
cycles barely move, you've moved work off the critical-path chunk.
Both are wins; they tell you different things.

### "Vendor parity" is the right default; "speed, not fidelity" is the override

`CLAUDE.md` Rule 6 sanctions deviating from vendor when faster. In
practice this session:

- **Vendor-parity changes that won.** All five landing commits in
  the final iteration batch were vendor-parity-shaped: LUT-fuse
  matches `DecodedData.hpp:316-337`; in-place ISA-L write matches
  vendor's `next_out` pointer pattern; constant-time CRC combine
  matches vendor's polynomial-multiply path. When in doubt, port
  vendor first; THEN measure.
- **One deliberate deviation worked.** Sync `with_sync_boundary_search`
  replaces vendor's parallel BlockFinder thread because measurement
  showed thread-spawn was 77 % of the call's elapsed time, dwarfing
  the streaming-overlap benefit. The deviation is documented at
  the call site with the stats that justified it.
- **One deviation regressed.** The branchless `replace_markers`
  attempt swapped the LUT load for a `v < 256` branch — looked
  faster on paper, branch mispredict cost beat the saved load.
  Reverted in 18 minutes.

The pattern: deviate when you have a per-call number showing the
deviation pays for itself. Don't deviate on aesthetic preference.

### Tools that earned their keep

- `make bench-sm` on neurotic — the authoritative wall-clock number.
- `perf record -F 999 --call-graph=dwarf` (with
  `RUSTFLAGS='-C debuginfo=1 -C strip=none -C force-frame-pointers=yes'`)
  — the only profile data worth optimizing against. Without symbols
  the report is hex addresses.
- `perf report --stdio --no-children -g graph,1,caller --group` —
  the right invocation. `--no-children` puts self-time on the leaf;
  `--group` combines E-core + P-core events on Intel hybrid; the
  `caller` view shows who's calling the hot symbol.
- `GZIPPY_DEBUG=1 gzippy -d -c -p 16 -v ...` — verbose decode stats
  (slow-path counts, BlockFinder spawn counts, bootstrap rate,
  buffer-pool hits/misses, prefetch metrics).
- Opus subagent for cross-checks (see above).

Tools that DIDN'T earn their keep this session:

- iai-callgrind for the inner Huffman loop. Synthetic inputs gave
  cycle-share numbers that didn't match silesia. Use real-workload
  perf instead.
- Mac arm64 local builds for x86-specific code. The
  `#![cfg(target_arch = "x86_64")]` gates mean local tests skip the
  failing tests; correctness verification has to run on neurotic.

### Stop when the surgical search is exhausted

After 5 wins and 3 reverts in one session, every remaining
high-percentage symbol in the profile is either (a) inside ISA-L's C
library, (b) in the kernel page-allocator path, or (c) already at
vendor parity. At that point the next move is structural — bigger
than a single commit — and should change the system's
shape, not its inner loops.

Signs you've reached this point:
- Two consecutive reverts.
- Top-5 hot symbols are all "vendor-parity already" or "can't touch".
- A 10 % cycle reduction translated to ~2 % wall improvement.
- CPU utilization across cores is < 50 % (workers idling, not
  saturating).

---

## 4. Opportunities for next

Ranked by evidence-strength for the win, with the surgical reverts
documented so they're not re-tried without new information.

### Status of the original S1–S4 candidates (2026-05-26 update)

- **S1 (madvise pre-fault) — FALSIFIED.** rpmalloc-per-Vec is already
  deployed via `arena-allocator` feature (`Cargo.toml:39, 51`;
  `rpmalloc_alloc.rs`). `clear_page_erms` at 17.8 % is at vendor parity
  (vendor ~17 %). The `0da3530` pre-allocate attempt regressed; remaining
  surgical surface is at-floor. Don't reattempt without daemon-mode
  amortization across invocations.
- **S2 (out-of-order drain) — FALSIFIED.** Writer must emit in order;
  CRC32 must combine in order. Out-of-order *completion* doesn't
  reduce wall because the slow chunk is on the critical path — it
  has to finish before its bytes are written. Memory cost real
  (~500 MB reorder buffer); benefit unclear.
- **S3 (global BlockBoundaryIndex) — FALSIFIED.** The current
  `with_sync_boundary_search` (`0abc483`) runs boundary scan
  per-chunk *in parallel* with decode. Doing it once up-front adds
  ~700 ms sequential CPU to the critical path (less if parallelized
  but then it's no longer "global, once").
- **S4 (segmented `chunk.data`) — PENDING.** Still real (~2 %
  cycles). Invasive (touches writer drain, `populate_subchunk_windows`,
  `get_last_window`); land only after the bigger-shape wins.

### Vendor-parity divergences from this session's archaeology

Read with: vendor file:line + gzippy file:line. Each labeled by
remaining surface area to land it.

- **CRC of resolved marker bytes — LANDED (`8ef7a4e`).** Vendor's
  `ChunkData::applyWindow` (`vendor/.../ChunkData.hpp:313-328`) CRCs
  the resolved marker bytes on the worker; we deferred to the consumer.
  Moved to `run_post_process_task` (`chunk_fetcher.rs:1672-1685`);
  ratio 0.53× → 0.62× (+17 % relative).
- **Proactive post-process pipelining — OPEN (medium surface).**
  Vendor `waitForReplacedMarkers` → `queuePrefetchedChunkPostProcessing`
  (`vendor/.../GzipChunkFetcher.hpp:521-551`) iterates the prefetch
  cache before blocking and submits post-process for every prefetched
  chunk whose predecessor window is published. Gzippy only submits
  post-process from the consumer's main loop (`chunk_fetcher.rs:1269`),
  so the post-process pipeline serializes through the consumer.
  Simpler shape (chain post-process inline at the end of
  `run_decode_task` when the predecessor window is already in
  `window_map`) has subtleties: `replace_markers_lut_narrow` doesn't
  clear `chunk.data_with_markers`, so the consumer's marker-vs-clean
  branch at `chunk_fetcher.rs:1119` would still take the marker path
  on an already-resolved chunk. Needs either a `chunk.post_processed`
  flag + new consumer branch, OR clear `data_with_markers` and
  refactor the tail-window publish to use `chunk.narrowed` for the
  trailing bytes.
- **`narrow_u16_to_u8` AVX2 disabled — OPEN (one env flip + verify).**
  AVX2 path at `chunk_fetcher.rs:1736` exists but is env-gated due to
  a suspected AVX2-license downclock through `apply_window`'s SIMD
  (`chunk_fetcher.rs:1707-1715`). Only matters for the non-fused
  path (small marker chunks < 128 KiB); silesia bench-sm never enters
  this path, so the win is for other workloads.
- **`publish_subchunk_windows` runs on consumer — OPEN (light surgical).**
  `chunk_fetcher.rs:2096`. Per-subchunk window_map insert serialized
  on consumer thread; could move to the worker right after
  `populate_subchunk_windows` (`chunk_fetcher.rs:1670`). Risk: ordering
  vs consumer's tail-window publish at `chunk_fetcher.rs:1130/1195`.
  Light work per chunk (only multi-subchunk chunks pay).
- **`insert_bytes_with_compression` allocates fresh `CompressedVector`
  per insert — OPEN (small).** `chunk_fetcher.rs:1130, 1195`. Each
  insert copies the 32 KiB tail into a new buffer. Vendor uses
  `shared_ptr<const CompressedVector>` and inserts the existing Arc.
  Could take `Arc<[u8; 32768]>` directly.
- **`ChunkData::clone` on `Arc::try_unwrap` miss — OPEN (verify-first).**
  `chunk_fetcher.rs:1086`. Diagnostic counter `ARC_TRY_UNWRAP_MISSES`
  tracks this; if non-zero in production, a per-chunk MB-scale clone
  sits on the consumer thread. Check the counter on a verbose run
  before acting.

### Open S4 candidate (re-stated)

#### S4. Segmented `chunk.data` to avoid `absorb_isal_tail`'s memmove

**Evidence.** ~2 % of cycles in the `dst.data.extend_from_slice(&tail.data)`
shift inside `absorb_isal_tail`. If `chunk.data` were `Vec<Vec<u8>>`
(segments), the tail could be moved by `Vec::append` (pointer swap)
instead of memcpy. Vendor stores data this way
(`std::vector<FasterVector<uint8_t>>`).

**Risk.** Downstream consumers (the writer drain, `populate_subchunk_windows`,
`get_last_window`) all assume contiguous `chunk.data`. Refactoring is
invasive.

### Surgical reverts to NOT re-try without new evidence

- **Pre-allocate `data_with_markers` to chunk capacity** (`0da3530`).
  Increased total cycles by 5 %. The page faults aren't from
  realloc; they're from first-touch on the new pool buffer.
  Pre-allocation doesn't help that.
- **Guard the hoisted `bits.refill()` with `bitsleft < 56`**
  (`d8e11a3`). Increased total cycles by 5 %. The branch broke
  instruction-level parallelism that the unconditional load+shift
  benefited from.
- **Branch `v < 256` instead of unified LUT in `replace_markers_lut_narrow`**
  (`63a1e97`). Increased `submit_post_process_to_pool` cycles
  from 6.1 % → 8.6 %. Branch mispredicts on mixed-distribution
  data dominated the saved LUT load.

If you have NEW evidence (e.g. a different fixture's distribution
makes the branch predictable, or `MADV_POPULATE_WRITE` shifts the
fault cost off-thread), re-try with a re-measurement plan. Don't
re-try because the source looks cleaner.

### Confirmed dead ends (don't even start)

- **SIMD marker scan in `emit_backref_ring`.** Maximum theoretical
  win < 1.5 % cycles on silesia (`plans/`-era abandoned plan).
  Below noise floor.
- **Hand-asm / inline-asm Huffman decode.** Outside the pure-Rust
  identity per `CLAUDE.md`'s deliverable definition.
- **Removing speculation entirely.** `GZIPPY_NO_PREFETCH=1` regressed
  bench-sm by +42 %. Speculation is net-positive despite the
  bootstrap cost it imposes on chunks whose `contains_marker_bytes`
  never flips.

---

## 5. Done-when criteria

Same as before. Run these four; if all four are green, the project is
done. If any one is red, the work items above say what's left.

1. `cargo bench --features pure-rust-inflate --bench inflate_isal_vs_pure_rust`
   reports **pure-Rust ≥ 1/1.05× ISA-L** on neurotic.
2. `cargo test --release --features pure-rust-inflate -- \
   test_single_member_parallel_silesia --ignored --nocapture` reports
   **ratio < 0.5** (T=16 parallel < ½ T=1 sequential) on neurotic.
3. `make ship` reports pure-rust-inflate throughput **≥ rapidgzip**
   throughput on the silesia corpus, same hardware.
4. `vendor/isa-l`, `vendor/isal-rs`, `packaging/isal-patches/`,
   `backends/isal_decompress.rs`, `backends/isal_compress.rs` are
   deleted; `arena-allocator` is decoupled from `isal-compression`
   in `Cargo.toml`; CI is green with `pure-rust-inflate` as the only
   inflate backend.

At the time of writing we are at ~0.53× rapidgzip with ISA-L still
the production path. None of the four gates are green. The
opportunities in §4 are the path from where we are to where the
gates close.
