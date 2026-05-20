# Parallel single-member correctness investigation (2026-05-20)

**Branch:** `feat/cross-chunk-retry`  
**HEAD at write time:** `c31d50f`  
**Audience:** External review (e.g. Opus) — self-contained; no prior chat context required.  
**Goal:** Document measured correctness state, root causes found, fixes applied, and what blocks rapidgzip-class **runtime** work.

---

## Executive summary

1. **Bisect pinned a deterministic correctness regression to `4909ac7`:** speculative prefetches (no predecessor window) were routed through `decode_chunk_isal_inexact` with an empty dictionary instead of the marker-bootstrap path. ISA-L cannot emit cross-chunk markers; it resolves unknown back-references against a zero-filled 32 KiB dict, producing silently wrong bytes → **CRC32 mismatch at T≥2**.

2. **Bug B is fixed on current HEAD** (`da84760` + `471a743`): prefetches again use `decode_chunk_marker_bootstrap_then_isal`, with minimal support modules restored (`deflate_block`, Huffman helpers, `isal_huffman`). **`make profile-decompression-x86_64` passes T=1 and T=2 reliably.**

3. **A separate failure remains at T≥4** on the same fixture: still `parallel SM: CRC32 mismatch`, reproducible on fresh x86 builds (not explained by Bug B). Consumer-only `WindowMap` publish (from `35ed3fb`) did not fix it; full cherry-pick of `35ed3fb` regressed further.

4. **Do not start rapidgzip throughput optimization until `profile-decompression-x86_64` is green for the full default sweep** (`T=1 2 4 8 9 12 16`). Perf work on a corrupt pipeline will waste time.

5. **A build footgun was found:** `da84760` restored `isal_huffman.rs` but omitted `pub mod isal_huffman` from `mod.rs`, so x86 release builds failed and the profile script exercised a **stale binary** — falsely implicating later commits.

---

## Test harness: `make profile-decompression-x86_64`

| Property | Value |
|----------|--------|
| Script | `scripts/profile_decompression_x86_64.sh` |
| Remote | `ssh -J neurotic root@10.30.0.199`, repo `/root/gzippy` |
| Fixture | 64 MiB raw (`benchmark_data/silesia-large.bin` head) → **`gzip -9 -c`** (GNU gzip CLI, not `gzippy -c`) |
| Reference | MD5 of **raw** source: `992bd4bace46842f4a6da435b9dc30e8` |
| Sweep | `GZIPPY_THREAD_SWEEP` default: `1 2 4 8 9 12 16` |
| Pass criterion | Every T: decode exit 0 **and** output MD5 == reference |
| Fail modes | `ERROR` = non-zero exit (often CRC mismatch); `WRONG` = exit 0 but MD5 differs |

**Why gzip(1) output:** Comments in the script state that only GNU `gzip(1)` layout reliably reproduces the production parallel-SM bug (as opposed to gzippy-compressed fixtures).

**Why x86 only:** `IsalParallelSM` is selected only when `isal_decompress::is_available()` (x86_64 + `isal-compression` feature) and `num_threads > 1` and compressed size > 10 MiB. arm64 dev machines never enter this path; local `cargo test` on Mac does not substitute.

**T=1 is a control, not the signal:** At T=1, routing selects `IsalSingle` (sequential ISA-L one-shot), not `chunk_fetcher::drive`. T=1 green only proves fixture + sequential path are sound.

---

## Production routing (what the harness exercises)

```
decompress_gzip_libdeflate
  → classify_gzip(data, num_threads)
  → IsalParallelSM  when: x86_64 + isal + T>1 + len > 10 MiB
  → single_member::decompress_parallel
  → sm_driver::read_parallel_sm
  → chunk_fetcher::drive
  → per-chunk decode (worker pool)
```

Relevant classifier (`src/decompress/mod.rs`):

- `num_threads > 1` && `data.len() > MIN_PARALLEL_COMPRESSED` (10 MiB) → `DecodePath::IsalParallelSM`
- else on x86 → `IsalSingle`

Verification failure is reported in `sm_driver::read_parallel_sm` as `ReadParallelSmError::CrcMismatch` → CLI `parallel SM: CRC32 mismatch` (no silent fallback; aligns with no-fallback policy).

---

## Worker decode routing (hot path)

`chunk_fetcher::run_decode_task` (`src/decompress/parallel/chunk_fetcher.rs`):

| Condition | Function | Window / dict |
|-----------|----------|----------------|
| `start_bit == 0` | `decode_chunk_isal_inexact` | 32 KiB zero window (chunk 0) |
| `window_map.get(start_bit)` hit | `decode_chunk_isal_inexact` | Materialized predecessor window |
| else (prefetch / slow path) | `speculative_decode_find_boundary` → `try_speculative_decode_candidate` | **Marker bootstrap** (no predecessor) |

**Fast path** (`decode_chunk_isal_inexact`): appends only to `chunk.data` via `append_owned_buffer` — **no `data_with_markers`**.

**Slow path** (`decode_chunk_marker_bootstrap_then_isal`): Phase 1 `deflate_block::Block` emits markers in `data_with_markers`; phase 2 raw ISA-L with clean 32 KiB dict for bulk `data`. Consumer later runs `apply_window` to resolve markers.

---

## Bisect narrative (pre-museum baseline → HEAD)

Original bisect table (from session; re-validated where noted):

| Commit | Subject | T=1 | T=2 | T=8 | Notes |
|--------|---------|-----|-----|-----|-------|
| `d7854e8` | fix(tests): typo seed | ok | ok | ERROR | |
| `32473b3` | Revert racy worker publish | ok | ok | ok* | *single run; see non-monotonic note |
| `581bf89` | trace-parity harness | ok | ok | ERROR | |
| `beff2fb` | trace-parity thread gate | ok | ok | ERROR | |
| `35ed3fb` | consumer-only window publish | ok | ok | ERROR | T=2 OK in bisect; see below |
| **`4909ac7`** | **prefetch → inexact ISA-L** | ok | **ERROR** | **ERROR** | **Introduced Bug B** |
| `791c1d1` | drop museum modules | ok | ERROR | ERROR | Inherits 4909ac7 |
| `89af21b` | Revert 35ed3fb | ok | ERROR | ERROR | Inherits 4909ac7 |
| `da84760` | restore marker bootstrap | ok | **ok** | ERROR | Bug B fixed |
| `471a743` | wire `isal_huffman` in mod.rs | ok | ok | ERROR | Build actually works |
| `c31d50f` | consumer-only publish (partial) | ok | ok | ERROR | T≥4 still fail |

**Lesson:** Clean OK→ERROR at T=2 across `4909ac7` is a **deterministic** signature. Reverting `35ed3fb` (`89af21b`) did not restore T=2 — bisect correctly blamed prefetch routing, not window publish.

---

## Bug B: empty-dict ISA-L on speculative prefetch (FIXED)

### Introduction — `4909ac7`

Commit message: route prefetches through `decode_chunk_isal_inexact` instead of marker bootstrap that failed with `InvalidHuffmanCode` on real gzip -9 boundaries.

`try_speculative_decode_candidate` (before fix):

```rust
let mut chunk = decode_chunk_isal_inexact(input, decode_start, until_bit, &[], configuration)?;
```

### Why this corrupts output

1. **`decode_chunk_isal_inexact` does not implement markers.** It only calls `chunk.append_owned_buffer(buffer)` — all output lands in `chunk.data` as clean bytes.

2. **`set_window(&[])` is not “unknown window”.** In `inflate_wrapper.rs`, empty slice maps to a **static 32 KiB zero dictionary**:

   ```rust
   let dict = if window.is_empty() { &ZERO_WINDOW[..] } else { window };
   ```

   The comment claims zeros are resolved later by `apply_window`; that is **false** for this path because nothing writes markers to `data_with_markers`.

3. **`apply_window` is a no-op** when `data_with_markers.is_empty()` (early return).

4. Cross-chunk back-references in the speculative chunk resolve against **zeros** (or wrong bytes), decode “succeeds”, stream CRC fails at end → `ReadParallelSmError::CrcMismatch`.

5. **Module doc in `gzip_chunk.rs` was wrong** after 4909ac7 (“empty initial_window emits markers”) — contradicted by implementation.

### Fix — `da84760` + `471a743`

- Restored `decode_chunk_marker_bootstrap_then_isal` and `bootstrap_with_deflate_block` in `gzip_chunk.rs` (from `4909ac7` tree before museum deletion).
- Restored **minimal** modules deleted in `791c1d1`:
  - `deflate_block.rs` (~2.3k LOC) — marker-emitting deflate port
  - `error.rs`, `gzip_definitions.rs`, `huffman_base.rs`, `huffman_symbols_per_length.rs`
  - `isal_huffman.rs` (x86, cfg-gated)
- `try_speculative_decode_candidate` calls `decode_chunk_marker_bootstrap_then_isal` again.
- **`471a743`:** added `#[cfg(...)] pub mod isal_huffman;` to `mod.rs` — without this, x86 `cargo build --release` fails with `cannot find isal_huffman in parallel`.

### Verified after Bug B fix

```
HEAD: da84760 / 471a743
T=1  992bd4ba…  ok
T=2  992bd4ba…  ok
T=4+ ERROR (CRC32 mismatch)  ← separate issue
```

---

## Build footgun: stale binary on failed compile

**Symptom:** After `da84760`, first `make profile-decompression-x86_64` run showed T=2 ERROR while commit message claimed bootstrap fix.

**Cause:** Remote build log:

```
error[E0433]: cannot find `isal_huffman` in `parallel`
```

Script continued to `test -x target/release/gzippy` — **old binary present** from previous successful build. Decodes used pre-`da84760` code (still broken prefetch routing).

**Fix:** `471a743` wires `isal_huffman` in `mod.rs`.

**Recommendation for harness:** Fail hard if `cargo build` errors (today stderr is grepped but script may still run decode sweep).

---

## Bug C (OPEN): T≥4 CRC mismatch on profile fixture

### Current measured state (fresh x86 builds, `c31d50f`)

Repeated runs on neurotic after `git reset --hard` + full rebuild:

| T | Result |
|---|--------|
| 1 | ok |
| 2 | ok |
| 3 | FAIL |
| 4 | FAIL |
| 5–16 | FAIL |

Default profile sweep (`T=1 2 4 8 9 12 16`) on `c31d50f`:

| T | Result |
|---|--------|
| 1, 2 | ok |
| 4, 8, 9, 12, 16 | ERROR (`parallel SM: CRC32 mismatch`) |

**Not reliably “T≥8 only”.** Earlier single profile runs showed T=4 **ok** once with a **cached** binary (build `Finished in 0.02s` without recompile) — treat sporadic T=4 pass as **stale-binary artifact**, not as fixed.

### What Bug C is not

- **Not Bug B:** T=2 is stable with marker bootstrap; empty-dict prefetch is not the explanation for T≥4 if T=2 passes on the same binary.
- **Not fully explained by “revert 35ed3fb”:** `35ed3fb` was reverted in `89af21b` for the wrong reason (T=2 bisect); re-applying window fixes has not greened T≥4.

### Window publish experiments (2026-05-20)

| Commit | Change | T=1 | T=2 | T=4 | T=8 |
|--------|--------|-----|-----|-----|-----|
| `471a743` | worker `insert_bytes` after decode | ok | ok | FAIL | FAIL |
| `e0f3e69` | full `35ed3fb` cherry-pick (+ partition `tryToDecode` at bootstrap name) | ok | ok | FAIL | FAIL |
| `c31d50f` | consumer-only publish only (no partition try) | ok | ok | FAIL | FAIL |

`35ed3fb` intended fix:

- Remove worker `window_map.insert_bytes` after `run_decode_task` (racy last-writer-wins).
- Publish fast-path tail windows on **consumer** when `data_with_markers.is_empty()`.

**Outcome:** T≥4 still fails; does not match “consumer-only fixes race” without further debugging.

### Hypotheses for Bug C (unverified; need trace)

1. **WindowMap key mismatch** — Consumer comments (~lines 788–802 in `chunk_fetcher.rs`) document a bug class: lookup must use `next_block_offset`, not `max_acceptable_start_bit` / chunk end. Wrong key → wrong window → wrong `apply_window` under higher parallelism.

2. **Marker path under concurrency** — Prefetch chunks have `data_with_markers`; consumer publishes tail at `chunk_end_bit` after `apply_window` in the marker branch. Ordering vs post-process queue (`pending`, `drain_one_pending`) may differ from vendor when T>2.

3. **Chain invariant / overshoot** — Documented vendor divergence in `gzip_chunk.rs` (lines ~297–333): gzippy’s `IsalInflateWrapper` does **not** cap `avail_in` at `until_bits`; workers stop at next EOB ≥ `until_bits`. `finalize(last_end_bit)` can publish encoded end > partition seed → prefetch guard rejects or wrong chunk chaining. Higher T may amplify on-demand decodes and expose ordering bugs.

4. **Post-process vs worker race** — Even with consumer-only publish for clean chunks, marker chunks still depend on predecessor window visibility via `window_map.contains` spin + post-process futures.

5. **Non-deterministic scheduling** — Less likely for **consistent** T=4 FAIL on repeated runs, but worth 10× runs after next fix.

### Suggested debug commands (neurotic)

```bash
cd gzippy && git checkout feat/cross-chunk-retry && cargo build --release --features isal-compression

# Reproduce
./target/release/gzippy -d -c -p 4 /tmp/profx86-fixture.gz > /tmp/out.bin 2>/tmp/err.txt
cat /tmp/err.txt

# Verbose route / stats
GZIPPY_VERBOSE=1 ./target/release/gzippy -d -c -p 4 /tmp/profx86-fixture.gz > /dev/null

# Trace (if GZIPPY_LOG_FILE supported)
GZIPPY_LOG_FILE=/tmp/sm-trace.jsonl ./target/release/gzippy -d -c -p 4 /tmp/profx86-fixture.gz > /dev/null
```

On branch machine (x86): `cargo test --release --features isal-compression --lib trace_parity` — includes gzip(1) CLI fixtures at multiple thread counts.

---

## Museum cleanup (`791c1d1`) — scope and impact

**Removed ~15k LOC** of unwired rapidgzip port files (`parallel_gzip_reader`, Huffman table variants, split `blockfinder_*`, etc.). **Production graph reduced to ~24 modules.**

**Re-added for bootstrap only** (`da84760`): `deflate_block` + minimal Huffman/error/`isal_huffman` — not a full museum restoration.

Docs updated: `docs/rapidgzip-port-reference.md` archived to short pointer; historical banners on marker-decoder docs.

**Perf implication:** Museum deletion does not fix or cause Bug B/C; it only removed dead code and accidentally removed bootstrap until restored.

---

## Performance context (blocked)

From `docs/baseline-2026-05-17.md` (homelab x86, silesia-large, T=16 decompress):

| Tool | MB/s (approx) |
|------|----------------|
| rapidgzip | ~1805 |
| gzippy (parallel SM, pre-fix era) | ~437–487 |

**Documented gzippy gaps (post-bootstrap, still relevant for perf planning):**

| Issue | Where documented | Effect |
|-------|------------------|--------|
| No `until_bits` / `avail_in` cap in wrapper | `gzip_chunk.rs` ~297–333 | Prefetch reject, pool efficiency ~28% vs vendor ~64% |
| Slow path cost | `chunk_fetcher.rs` comments | Block finder + marker bootstrap per candidate |
| Consumer serialization | `PARALLEL_PROFILING_PLAN.md` | Tmax loss often off-CPU (waits), not hot functions |

**Do not optimize until:** `profile-decompression-x86_64` full sweep PASS.

**Perf priority after green profile (one change per commit):**

1. P0: Port `m_encodedUntilOffset` into `IsalInflateWrapper::refill_buffer` (chain invariant + prefetch reuse).
2. P1: `bench-sm` / homelab table — gzippy vs rapidgzip on silesia, T=1 and T=16.
3. P2: `GZIPPY_LOG_FILE` + `scripts/timeline.py` (consumer wait vs worker idle).
4. P3: Prefetch hit rate (block finder + partition seed; avoid scanning all candidates).
5. P4: Maximize fast-path `decode_chunk_isal_inexact` share when window known.

---

## Commit reference (newest first)

| SHA | Summary |
|-----|---------|
| `c31d50f` | Consumer-only WindowMap publish (no partition tryToDecode) |
| `e0f3e69` | Full 35ed3fb cherry-pick (superseded by c31d50f intent) |
| `471a743` | Wire `isal_huffman` in `mod.rs` |
| `da84760` | Restore marker bootstrap for speculative prefetch |
| `89af21b` | Revert 35ed3fb |
| `791c1d1` | Drop museum modules; archive docs |
| `4909ac7` | **Bug B introduced** — prefetch → inexact ISA-L |
| `35ed3fb` | Consumer-only window publish (still in history) |

---

## Invariants and policies (review checklist)

- [ ] No fallback on parallel SM failure — CRC mismatch is hard error (correct).
- [ ] Speculative prefetch **must** use marker bootstrap when predecessor window absent.
- [ ] `decode_chunk_isal_inexact` **must not** be documented as emitting markers.
- [ ] x86 release build must include `isal_huffman` when `deflate_block` is compiled.
- [ ] Profile harness must not run decodes if build fails.
- [ ] T≥4 correctness on gzip(1) fixture before any perf claim.

---

## Recommended next steps (for reviewer / implementer)

### Immediate (correctness)

1. **Confirm Bug C on `c31d50f`** with fresh build + full thread sweep; capture `GZIPPY_VERBOSE` stderr for first failing T.
2. **Run `trace_parity` on x86** — may cover gzip(1) T=16 even when profile fails at T=4.
3. **Audit WindowMap keys** on marker vs fast-path chunks under T=4 — compare to vendor `GzipChunkFetcher.hpp` post-process publish points.
4. **Consider binary search inside `chunk_fetcher::consumer_loop`** with trace events at window insert/get.

### After profile PASS

5. Implement wrapper `until_bits` cap (P0 perf + may reduce ordering bugs).
6. Establish bench-sm baseline vs rapidgzip.
7. Cloud fleet / `make ship` only after local profile + bench-sm stable.

### Do not

- Re-import full museum.
- Route prefetches through empty-dict ISA-L “for speed”.
- Trust arm64 `cargo test` alone for parallel SM.
- Trust profile results when remote `cargo build` shows errors.

---

## Appendix A: Key source locations

| Topic | File | Symbols / lines |
|-------|------|-----------------|
| Routing | `src/decompress/mod.rs` | `classify_gzip`, `DecodePath::IsalParallelSM` |
| Entry | `src/decompress/parallel/sm_driver.rs` | `read_parallel_sm`, CRC verify |
| Orchestration | `src/decompress/parallel/chunk_fetcher.rs` | `drive`, `consumer_loop`, `run_decode_task` |
| Fast decode | `src/decompress/parallel/gzip_chunk.rs` | `decode_chunk_isal_inexact` |
| Slow decode | `src/decompress/parallel/gzip_chunk.rs` | `decode_chunk_marker_bootstrap_then_isal`, `bootstrap_with_deflate_block` |
| Markers | `src/decompress/parallel/deflate_block.rs` | `Block::read`, marker ring |
| Window apply | `src/decompress/parallel/apply_window.rs` | `apply_window` |
| Empty dict | `src/decompress/parallel/inflate_wrapper.rs` | `set_window` |
| Profile script | `scripts/profile_decompression_x86_64.sh` | full harness |

---

## Appendix B: Original bisect table (verbatim reference)

```
reference (correct decode) = 992bd4bace46842f4a6da435b9dc30e8

| Commit  | T=1 | T=2          | T=8          | Subject                                      |
|---------|-----|--------------|--------------|----------------------------------------------|
| d7854e8 | ok  | ok           | ERROR        | fix(tests): typo 0xb0undary                   |
| 32473b3 | ok  | ok           | ok           | Revert "remove racy worker publish"          |
| 581bf89 | ok  | ok           | ERROR        | chore(trace-parity): harness                 |
| beff2fb | ok  | ok           | ERROR        | test(trace-parity): thread-independence gate |
| 35ed3fb | ok  | ok           | ERROR        | fix: consumer-only window publish            |
| 4909ac7 | ok  | ERROR        | ERROR        | fix: speculative prefetch via inexact ISA-L  |
| 791c1d1 | ok  | ERROR        | ERROR        | chore: drop museum modules                   |
| 89af21b | ok  | ERROR        | ERROR        | Revert "consumer-only window publish"        |
```

Post-investigation additions: `da84760` fixes T=2; `471a743` fixes build; T≥4 remains open on current HEAD.

---

## Appendix C: Bug C is NON-deterministic — a race (2026-05-20, later)

**HEAD:** `2ab290c` (= `c31d50f` + commit `2ab290c` which adds the
`GZIPPY_WINTRACE` diagnostic; no production-logic change).

### Finding 1 — Bug C is a race, not a deterministic bug

The body of this document calls Bug C "deterministic" (§"Bug C", and
"Clean OK→ERROR cutover = deterministic signature"). **That is wrong.**
Measured on neurotic, fresh build, gzip(1) 64 MiB fixture, `-p 4`:

```
6 plain T=4 runs:  run1 ok  run2 ok  run3 ERROR  run4 ERROR  run5 ERROR  run6 ERROR
```

2 pass / 4 fail on the *same binary, same input*. Bug C is
**non-deterministic**. The earlier "consistent T=4 FAIL" was an unlucky
run of consecutive failures — a single decode has roughly a 1-in-3
chance of passing, so a short streak of either outcome is easy to hit.

**Consequence for the harness:** `make profile-decompression-x86_64`
decodes each T **once**. At T=4 that is a ~33% false-pass rate. The
harness must loop each thread count ≥10× and fail if *any* run fails,
before its PASS means anything.

### Finding 2 — `eprintln` tracing masks the race (Heisenbug)

`GZIPPY_WINTRACE=1` (commit `2ab290c`) traces every WindowMap
insert/get via `eprintln!`. With it on:

```
3 traced T=4 runs:  3/3 PASS
```

`eprintln!` to stderr takes a process-global lock; ~228 trace lines
serialize the worker threads enough that the race stops firing. **Any
future tracer for this bug must be low-perturbation** — a pre-allocated,
lock-free event array (atomic index, fixed-size, Copy structs), dumped
once at process exit. No I/O, no locks during decode.

### Finding 3 — the collision is visible even in a passing run

The masked (passing) traced run still shows the mechanism. WindowMap
key `33700378` is inserted **twice with different content**:

```
[WIN] insert key=33700378  fp=b2f7856a5dd4a26d
[WIN] get    key=33700378  HIT  fp=b2f7856a5dd4a26d
   ... (many inserts later) ...
[WIN] insert key=33700378  fp=102b629884ee3096     ← same key, different bytes
```

(`fp` = FNV-1a of the 32 KiB window; equal fp ⇔ identical bytes.) A
`get` consumed the *first* value; a later `insert` overwrites with a
*different* window. On a passing run the consumer read the right one;
on a failing run it reads the wrong one. That is the whole bug.

Also seen: key `30993014` and key `17175561` (chunk 0's tail) carry the
**same** fp `1959a4450f8aad76` — two unrelated keys, identical window
bytes. Either repeated source data, or a misdirected publish; unresolved.

### Finding 4 — why the collision exists (mechanism; partly hypothesis)

From the trace, speculative prefetch chunks **overshoot and overlap**:

```
[WIN] decode start=16777216 ... slow-spec -> enc=[16777216, 33700378)
[WIN] decode start=33554432 ... slow-spec -> enc=[33554432, 50347531)
```

`[16777216,33700378)` and `[33554432,50347531)` overlap on
`[33554432,33700378)`. Two chunks decode the same bit range.

Window publish sites (`chunk_fetcher.rs`):
- consumer fast-path branch — `insert_bytes(chunk_end_bit, tail)` (~754)
- consumer marker-path branch — `insert_bytes(chunk_end_bit, tail)` (~818)
- post-process per-subchunk — `insert_bytes(sc_end_bit, w)` (~1179)

The last subchunk's `sc_end_bit` equals the chunk's `chunk_end_bit`, so
**every chunk's end-key is written by at least two paths** (the consumer
AND that chunk's post-process). Add cross-chunk overshoot/overlap and a
single bit-offset key can receive windows computed from *different
decode histories* — one chunk that decoded the full preceding 32 KiB,
another that started mid-region and only has a partial/zero-seeded
window. Last-writer-wins (`WindowMap` uses `insert_or_assign`). Which
writer lands last is scheduling-dependent ⇒ race.

**Hypothesis (needs the low-perturbation trace of a FAILING run to
confirm):** a window key K is written both by the authoritative chunk
for K and by an overlapping speculative chunk; when the speculative
(wrong, partially zero-seeded) window wins, the chunk starting at K gets
a corrupt dictionary → wrong bytes → end-of-stream CRC mismatch. T≤2
mostly dodges it because prefetch depth is too shallow to create
overlapping speculative chunks; T≥3 creates enough overlap to race.

### Finding 5 — Bug A and Bug C are very likely the same bug

This document treats Bug A ("T≥8 racy window-publish") and Bug C ("T≥4
deterministic CRC mismatch") as separate. Findings 1–4 indicate they are
**one bug**: a window-publish key collision between overlapping/
speculative chunks. Bug C only looked deterministic because of the
short-run sampling error in Finding 1.

### What is NOT yet known

- Whether the post-process per-subchunk publish runs for chunks whose
  speculative decode the consumer later *re-anchored* — i.e. whether the
  colliding writer is a stale speculative chunk or a legitimate one.
- The exact key and the two windows on a *failing* run (the trace above
  is from a passing run).

### Corrected next step

1. Replace the `eprintln` tracer with a lock-free fixed-size event array
   (atomic-index, Copy structs, dump at exit). Low enough perturbation
   that the race still fires under trace.
2. Capture a **failing** T=4 run. Identify the first key whose `get`
   returns a window whose fp ≠ the fp the authoritative chunk wrote.
3. Fix the publish: one window per key, written only by the chunk
   authoritative for that boundary (vendor publishes from the in-order
   consumer on accepted chunks only — never from speculative workers,
   and accepted chunks do not overlap).
4. Make `make profile-decompression-x86_64` loop each T ≥10×.

---

*End of investigation document. Appendix C added 2026-05-20.*
