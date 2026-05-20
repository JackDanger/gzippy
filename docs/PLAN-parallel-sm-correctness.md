# Plan: parallel single-member correctness (x86 / ISA-L)

**Branch:** `feat/cross-chunk-retry`  
**Gate:** `make profile-decompression-x86_64` PASS — no perf claims before this.

---

## Goal

Byte-perfect decode of GNU `gzip -9` single-member fixtures on x86 at  
`T ∈ {1, 2, 4, 8, 9, 12, 16}`, then pursue rapidgzip-class throughput.

---

## Current state

| Item | Status |
|------|--------|
| **Prefetch empty-dict bug** (`4909ac7`) | **Fixed** — `da84760` / `471a743` restore `decode_chunk_marker_bootstrap_then_isal`; `isal_huffman` wired in `mod.rs` |
| **T=1, T=2** on profile fixture | **Reliable PASS** (fresh x86 build) |
| **T≥3** on same fixture | **Flaky FAIL** — `parallel SM: CRC32 mismatch` (~33% pass at T=4) |
| **Window publish race** | **Open** — same root cause as earlier “Bug A” / “Bug C” labels |
| **“Consumer-only publish”** (`35ed3fb` / `c31d50f`) | **Did not green profile** — see Phase 2; workers were narrowed but post-process still publishes on the pool |

T=1 is a control only (`IsalSingle`, not `chunk_fetcher::drive`).

---

## What we know (facts)

1. **Non-deterministic race**, not a deterministic T≥4 bug. Six plain T=4 runs: 2 pass, 4 ERROR (same binary + input).

2. **`eprintln` tracing masks it** (`GZIPPY_WINTRACE`): stderr lock serializes workers; 3/3 traced runs pass. Any tracer must be lock-free and I/O-free during decode.

3. **Collision mechanism** (seen even in passing traced runs): same `WindowMap` key (e.g. `33700378`) inserted twice with different 32 KiB fingerprints; `get` may consume the first while a later `insert` overwrites. Last-writer-wins + scheduling ⇒ flaky CRC.

4. **Overlap**: speculative slow-path chunks overshoot partition seeds; encoded ranges overlap (e.g. `[16777216,33700378)` vs `[33554432,50347531)` share bits). Multiple publish sites: consumer fast/marker branches, post-process per-subchunk — chunk end key often written twice.

5. **“Consumer-only” was incomplete**: worker `insert_bytes` after decode was removed, but `run_post_process_task` still calls `window_map.insert_bytes(sc_end_bit, …)` on the thread pool (`submit_post_process_to_pool` → `thread_pool.submit`, priority −1). That is a **concurrent second publisher** — why `35ed3fb` / `c31d50f` did not fix the race.

6. **Build footgun**: failed x86 build + stale `target/release/gzippy` produced false bisect/profile results. Harness must abort if `cargo build` errors.

**Types vs this session’s meta-bugs:** Rust types help with conflation and aliasing (wrong key kind, publish from wrong phase). They do **not** fix scheduling order, wrong fixtures (`gzippy -c` vs `gzip(1)`), or acting on unmeasured hypotheses. Phase 0 (harness) and Phase 1 (lock-free trace of a **failing** run) are the antidotes to race-debugging flailing — not Phase 3 in advance.

**Compile time cannot fix scheduling.** Types can make “a second writer exists” unrepresentable after Phase 2 fixes the architecture — not “types fix the race.”

---

## Harness (`scripts/profile_decompression_x86_64.sh`)

| Property | Value |
|----------|--------|
| Remote | `ssh -J neurotic root@10.30.0.199`, repo `/root/gzippy` |
| Fixture | 64 MiB raw → **`gzip -9 -c`** (not gzippy-compressed) |
| Reference | raw MD5 `992bd4bace46842f4a6da435b9dc30e8` |
| Path | `IsalParallelSM` → `sm_driver::read_parallel_sm` → `chunk_fetcher::drive` |

**Required harness changes (Phase 0):**

- Loop each thread count **≥10×**; FAIL if any trial fails.
- Fail hard when remote `cargo build` fails (do not run decode on stale binary).
- Optional: env `GZIPPY_PROFILE_TRIALS` default `10`.

---

## Phase 0 — Harness & build hygiene

- [ ] Multi-trial decode loop per `T` in profile script.
- [ ] Treat any `cargo build` error as `BUILD FAILED` (no decode sweep).
- [ ] Document pass criterion in script header (flaky single-shot was ~33% false-pass at T=4).

---

## Phase 1 — Find failing race under trace

**Do Phase 1 before Phase 3.** Measure the invariant on a failing run; encode it in types only after Phase 2 works. Type design is more pleasant than race-debugging — that pull is a trap.

- [ ] Replace `GZIPPY_WINTRACE` `eprintln` with fixed-size lock-free event buffer (atomic index, `Copy` events, dump at exit).
- [ ] Capture a **failing** T=4 run; find first `get(key)` where fingerprint ≠ authoritative insert for that key.
- [ ] Optional: `cargo test --release --features isal-compression --lib trace_parity` on x86.

---

## Phase 2 — Fix publish protocol (correctness)

**Invariant:** at most one authoritative window per boundary; **one thread** owns all inserts into the shared map (the in-order consumer).

The race is two legal `insert` calls in the wrong order — types cannot order them. Phase 2 removes the second writer; Phase 3 makes that layout non-bypassable.

- [x] **`run_post_process_task` per-subchunk publish** moved to consumer `publish_subchunk_windows` in `drain_one_pending` (vendor `appendSubchunksToIndexes`). Pool task is `apply_window` only; tail emplace uses `CompressionType::None`.
- [ ] Remove all other worker-side `WindowMap` inserts (including any leftover worker publish after `run_decode_task`).
- [ ] Consumer sole publisher for authoritative boundaries; no concurrent post-process inserts.
- [ ] Stop publishing windows from overlapping speculative encoded ranges; align `until_bits` / `avail_in` cap with vendor (`gzip_chunk.rs` / `inflate_wrapper.rs`).
- [ ] Audit keys: consumer must `get`/`publish` at `next_block_offset`, not chunk end / `max_acceptable_start_bit` conflation.
- [ ] Re-run profile with ≥10 trials per T until green.

Optional after the above is green: `try_publish` → loud error on double insert (runtime guard, not prevention; redundant on a single-threaded map).

**Do not:** route prefetches through `decode_chunk_isal_inexact` + empty dict; re-import museum modules; trust arm64 `cargo test` for this path.

---

## Phase 3 — Encode invariants in types (after Phase 2 green)

**Sequence:** Phase 1 trace → Phase 2 fix → Phase 3 enforce. Do not design types from speculation.

Frame: types make the **fixed** architecture non-bypassable; they do not replace measurement or fix scheduling.

### Bucket A — Race fix enforcement (same change as Phase 2, viewed as types)

Ship with Phase 2, not a separate “typing” milestone.

- [ ] **`WindowMap` consumer-only** (`!Sync` or owned only by consumer task; workers never hold `&mut WindowMap`). Makes worker/post-process publish **unrepresentable**. Also drops `Mutex<BTreeMap<…>>` around the map → **strictly faster**, not just safer.
- [ ] **`SpeculativeChunk` vs `CommittedChunk`** (optional belt-and-suspenders): only `CommittedChunk` may be published; consumer can still hold speculative data during re-anchoring. Zero cost once map is consumer-only.

### Bucket B — Marker bug class (separate PR, after profile green)

- [ ] **`ChunkData<UnresolvedMarkers>` → `apply_window` → `ChunkData<Clean>`** — prevents writing unresolved marker output. Different bug class than the race; phantom-type churn but real value.

### Bucket C — Bit-offset conflation (separate PR, lowest urgency)

- [ ] **`PartitionSeed` / `BlockBoundary` / `ChunkEnd`** newtypes (`#[repr(transparent)]`, zero cost) — `next_block_offset` vs `max_acceptable_start_bit` vs chunk end. Large mechanical refactor; targets conflation, not the race (mostly tamed by renames already).

**Cut from plan (do not build speculatively):** `Window { provenance }` debug asserts; `PrefetchHandoff` / `PrefetchReuseToken`; `EncodedRange::overlaps()` unless sharpened to “committed chunk encoded ranges must not overlap” (speculative overlap is intentional).

---

## Phase 4 — Performance (blocked until Phase 2 green)

Do not tune throughput on a corrupt pipeline.

| Priority | Work |
|----------|------|
| P0 | `with_until_bits` wired in fast + slow ISA-L decode (wrapper `refill_buffer` cap) |
| P1 | `bench-sm` vs rapidgzip (silesia, T=1, T=16) |
| P2 | `GZIPPY_LOG_FILE` + timeline (consumer wait vs worker idle) |
| P3 | Prefetch hit rate (block finder + partition seed) |
| P4 | Maximize fast-path `decode_chunk_isal_inexact` when window known |

Reference throughput gap (homelab, pre-fix era): rapidgzip ~1805 MB/s vs gzippy ~437–487 MB/s at T=16 — reclaim only after correctness gate.

---

## Worker decode routing (must hold)

| Condition | Function |
|-----------|----------|
| `start_bit == 0` or `window_map.get(start_bit)` | `decode_chunk_isal_inexact` (known window) |
| else (prefetch) | `speculative_decode_find_boundary` → `decode_chunk_marker_bootstrap_then_isal` |

Fast path: no markers. Slow path: markers in `data_with_markers`, then ISA-L bulk into `data`; consumer `apply_window`.

---

## Key source locations

| Topic | File |
|-------|------|
| Routing | `src/decompress/mod.rs` |
| Entry + CRC | `src/decompress/parallel/sm_driver.rs` |
| Orchestration / post-process publish | `src/decompress/parallel/chunk_fetcher.rs` (~1179, ~1008 pool submit) |
| Fast / slow decode | `src/decompress/parallel/gzip_chunk.rs` |
| Markers | `src/decompress/parallel/deflate_block.rs` |
| Window map | `src/decompress/parallel/window_map.rs` |
| Empty dict | `src/decompress/parallel/inflate_wrapper.rs` |
| Profile script | `scripts/profile_decompression_x86_64.sh` |

---

## Commit reference (correctness arc)

| SHA | Note |
|-----|------|
| `4909ac7` | Introduced empty-dict prefetch (Bug B) |
| `da84760` / `471a743` | Marker bootstrap + `isal_huffman` mod |
| `35ed3fb` / `c31d50f` | Incomplete “consumer-only” (post-process still on pool) |
| `2ab290c` | `GZIPPY_WINTRACE` diagnostic (masks race) |

---

## Policies

- No fallback on parallel SM failure — CRC mismatch is hard error (correct).
- Speculative prefetch **must** use marker bootstrap when predecessor window absent.
- x86 release build must compile `isal_huffman` when `deflate_block` is enabled.
- Profile PASS means **all trials** at **all** sweep thread counts.
