# Path-mix instrumentation (Gate 0)

**Goal:** Compare decode **path mix** between gzippy and rapidgzip on the same corpus — not wall time.

**Reference runs:**
- `0462f88`: gzippy 90% window-absent / 97% KEY-MISMATCH.
- `738dea6`: gzippy unchanged; `bad_seed_resync=0`; rapidgzip `worker.decode` spans=0 (stale binary — guest script fixed in `f8490b3` to always apply patches + rebuild when span missing).

---

## Per-decode-task (both tools must emit)

Keyed by `start_bit`:

| Field | Meaning |
|-------|---------|
| `mode` | `clean` \| `boundary_search` / `window_absent` |
| `window_present` | `WindowMap::get(start)` hit |
| `window_exact` | chain-exact stop hint |
| `speculative` | prefetch vs on-demand |

**gzippy:** `causal.decode_decision`, `worker.decode_mode` (`chunk_fetcher.rs` `run_decode_task`).

**rapidgzip:** `scripts/rapidgzip_trace_patch/patch_model_params.sh` — `worker.decode` span with mode after `sharedWindow` lookup (`GzipChunkFetcher.hpp` ~712).

---

## Per tryToDecode attempt (both tools)

| Field | Meaning |
|-------|---------|
| `partition_seed` | `offset.first` |
| `decode_start` | `offset.second` (seek bit) |
| `ok` | success |
| `fail_kind` | `header` \| `body` \| `inflate` \| `stop_missed` \| `other` |

**gzippy (Gate 0 landed):** `worker.try_to_decode` instant in `try_speculative_decode_candidate`.

**rapidgzip:** patch `GzipChunk.hpp` tryToDecode lambda catch/success with same instant shape.

---

## Per-chunk phase outcome (gzippy)

`worker.chunk_phase` instant:

| `phase` | When |
|---------|------|
| `window_seeded` | Full 32 KiB window at chunk start |
| `flip_to_clean` | Marker bootstrap → `finish_decode` handoff |
| `finished_no_flip` | BFINAL / stop-hint without flip |

**Verbose aggregates (`GZIPPY_VERBOSE`):** `flip_to_clean`, `finished_no_flip`, `finish_decode`, `inflate_wrapper`, `window_seeded`, `bad_seed_resync` (must stay 0 post–Gate 1).

---

## Gate 1 change (vendor tryToDecode semantics)

Removed `bad_seed_resync` fallback in `decode_chunk_with_rapidgzip_impl`. Bootstrap failure **propagates**; `speculative_decode_find_boundary` catches and tries next candidate — matches vendor `catch` + continue.

**Regression check:** `bad_seed_resync == 0` on any production run.

---

## Comparison workflow

1. Run locked Fulcrum with `GZIPPY_VERBOSE=1` and `GZIPPY_TIMELINE` on both tools.
2. Count `worker.try_to_decode` ok/fail by `fail_kind` per tool.
3. Count `worker.chunk_phase` histogram per tool.
4. Join on `start_bit` with `causal.decode_decision` for window hit/miss duty cycle.
