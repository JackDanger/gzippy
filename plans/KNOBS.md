# KNOBS.md — the GZIPPY_* env-knob / oracle index (P2 of OWNER-HELP-SUGGESTIONS)

A standing index so a misread knob can never cause a wrong-premise turn. Regenerate
the skeleton with `rg -o 'GZIPPY_[A-Z_0-9]+' src/ | sort -u`; this file curates the
semantics + CLASS the bare list cannot give you.

## How to read CLASS (ties to the Measurement PROCESS in CLAUDE.md)

- **perturbation** — SLOWS a region by a known factor to test whether it gates the
  wall (PROCESS rule 1). A monotonic interleaved wall response ⇒ critical; flat ⇒
  slack. ALWAYS pair a busy-spin with a frequency-neutral SLEEP control (rule 2).
  A slow-down slope is NOT a speed-up ceiling (rule 3).
- **oracle** — REMOVES a region to set a SPEED-UP CEILING (rule 3). Read the
  oracle's self-reported overhead and REFUSE a contaminated contender (fulcrum_total
  guarantee 4). Never extrapolate a slow-down slope through an unlocated knee — REMOVE
  and measure.
- **instrument** — trace/counter/log. OFF==identity (must not change the hot path).
  Validate the instrument before trusting it (rule 4): positive/negative controls,
  self-test reads 1.0 ± spread.
- **behavior** — changes the production PATH / a routing or buffering choice. Use to
  A/B a structural decision, NOT as a perturbation or a ceiling.
- **routing-assert** — used to ASSERT the production path (e.g. force the SM engine),
  not to change perf semantics.
- **build** / **test** — affects the build feature set or test harness only.

> WINDOW-ABSENT-PRESERVING is the cardinal rule: any knob that SEEDS windows or
> swaps the engine routes off the production (window-absent marker bootstrap) path
> and MASKS the binder. Such a run is NOT production — `parity.sh` and
> `fulcrum_total` both REFUSE to call it one. Those knobs are tagged **(masks-binder)**.

---

## Index

| Knob | Read-site | CLASS | What it does |
|------|-----------|-------|--------------|
| `GZIPPY_DEBUG` | src/utils.rs:8 | instrument | Cached debug flag; with `-d` prints `path=…` so you can assert `ParallelSM` (the production routing-assert). |
| `GZIPPY_FORCE_PARALLEL_SM` | (read across parallel/) | routing-assert | Forces the parallel single-member engine at any T so the production path is exercised even below the size gate. The 323-use workhorse. |
| `GZIPPY_VERBOSE` | src/main.rs:186 | instrument | Dumps the counter sidecar to stderr (header/body counts, `window_seeded`, flip/no-flip, oracle state). The sidecar fulcrum_total reads to verify window-absent routing. |
| `GZIPPY_TIMELINE` | src/decompress/parallel/trace_v2.rs:1 | instrument | Writes the Chrome-trace timeline to the given file for fulcrum_total. OFF==identity. |
| `GZIPPY_TRACE_DETAIL` | trace_v2.rs:69 | instrument | Extra per-span detail in the timeline; off the hot path unless =1. |
| `GZIPPY_TRACE` | src/decompress/bgzf.rs:77 | instrument | bgzf-path detailed perf breakdown to stderr. |
| `GZIPPY_TRACE_DRAIN` | chunk_fetcher.rs:3787 | instrument | Traces the drain/consumer path. |
| `GZIPPY_LOG_FILE` | src/decompress/parallel/trace.rs:5 | instrument | When set, logs every SM lifecycle event to a file (analyze with scripts/parallel_sm_log_summary.py). |
| `GZIPPY_PID` | trace_v2.rs:35 | instrument | Tags trace events with a pid (multi-process trace disambiguation). |
| `GZIPPY_MEMLIFE` | src/decompress/parallel/memlife.rs:2 | instrument | On-demand memory-lifetime instrumentation. |
| `GZIPPY_MEM_STATS` | src/main.rs:119 | instrument | Debug memory accounting; no-op unless set. |
| `GZIPPY_RPMALLOC_STATS` | rpmalloc_alloc.rs:496 | instrument | Dump rpmalloc allocator stats; normal runs pay nothing. |
| `GZIPPY_STORED_PHASE_TIMING` | stored_split.rs:52 | instrument | Prints per-phase wall of the stored-block split path. |
| `GZIPPY_STALL_RESIDENCY_PROBE` | stall_residency.rs:12 | instrument | Residency/stall probe; OFF is an inlined early-return (OFF==identity). |
| `GZIPPY_SLOW_HITS` | src/main.rs:122 | instrument | Counts slow-injection hits — the instrument-validity proof that a perturbation actually fired. |
| `GZIPPY_SLOW_MODE` | slow_knob.rs:14 | perturbation | Numeric percent string: `"25"` ⇒ +25% injected into the targeted region. The primary criticality dial (rule 1). |
| `GZIPPY_SLOW_KIND` | slow_knob.rs:17 | perturbation | `"spin"` (busy ALU, default) vs `"sleep"` (yields the core) — the frequency-neutral control (rule 2). |
| `GZIPPY_SLOW_DECODE` | slow_knob.rs:167 | perturbation | Injects ONLY at Huffman-decode events (isolates the inner loop's criticality). |
| `GZIPPY_SLOW_STORE` | slow_knob.rs:171 | perturbation | Injects ONLY at literal-store / back-ref-copy events. |
| `GZIPPY_SLOW_MARKER_MODE` | slow_knob.rs:98 | perturbation | Marker-mode slow factor (percent/100); `0.0` ⇒ OFF. Tests the marker-resolve term. |
| `GZIPPY_SLEEP_DECODE_NS` | decode_bypass.rs:259 / chunk_fetcher.rs:2378 | perturbation | Fixed per-chunk sleep (coordination-isolation); used with bypass capture to project a target decode rate. (masks-binder when combined with bypass — output is garbage.) |
| `GZIPPY_BYPASS_CAPTURE` | chunk_fetcher.rs:2549 | oracle | CAPTURE pass: a NORMAL decode that records per-chunk results to a file for later replay. |
| `GZIPPY_BYPASS_DECODE` | chunk_fetcher.rs:631 | oracle | REPLAY pass: worker decode is replaced by a file read (decode≈0) → the FLOOR oracle (preserves form-B markers ⇒ L_resolve still runs). Misses fall back to real decode (keeps bytes correct, inflates the floor). |
| `GZIPPY_BYPASS_META_ONLY` | decode_bypass.rs:105 | oracle | Capture only metadata (tiny file) for the sleep-probe variants (resolve-elided ⇒ garbage out, NOT a floor). |
| `GZIPPY_BYPASS_FORCE_CLEAN` | decode_bypass.rs:45 | oracle | Form-A-only replay mode. (masks-binder — elides marker resolve.) |
| `GZIPPY_BYPASS_REBUILD` | decode_bypass.rs:457 | oracle | Use the per-call rebuild path in the bypass chain (A/B of the replay machinery). |
| `GZIPPY_SEED_WINDOWS` | perfect_overlap.rs:34 | oracle | **(masks-binder)** Seedfull: pre-seeds chunk windows so the marker bootstrap never runs → routes to the CLEAN engine. A SEEDED run is a CEILING, never production. |
| `GZIPPY_SEED_WINDOWS_CAPTURE` | seed_windows.rs:19 | oracle | **(masks-binder)** Captures the seed windows for a later seeded run. |
| `GZIPPY_SEED_NO_WINDOWS` | seed_windows.rs:104 | oracle | **(masks-binder)** DECOMPOSE: suppress the window seed contribution (measurement-only). |
| `GZIPPY_SEED_NO_BOUNDARIES` | chunk_fetcher.rs:500 | oracle | DECOMPOSE: skip the boundary-seed step (measurement-only). |
| `GZIPPY_CLEAN_WINDOW_ORACLE` | sm_driver.rs:43 | oracle | **(masks-binder)** Decode using a clean-window oracle (default OFF). This is the instrument class that was PROVEN BROKEN (silently re-ran the bootstrap, 64eb6df) — distrust by default; validate before use. |
| `GZIPPY_ISAL_ENGINE_ORACLE` | gzip_chunk.rs:130 | oracle | **(masks-binder)** Swap the chunk engine to ISA-L instead of pure-Rust to prove an engine-speed ceiling. NOT production (C-FFI is off the decode graph). |
| `GZIPPY_PERFECT_OVERLAP` | perfect_overlap.rs:3 | oracle | PERFECT-OVERLAP removal oracle: removes the publish-chain serialization to bound the overlap ceiling. |
| `GZIPPY_SKIP_WRITEV_SYSCALL` | output_writer.rs:8 | oracle | Removal oracle for the writev syscall (validated: it is the output-cost term). OFF==identity. |
| `GZIPPY_DECODE_FREE` | (not in current src) | oracle | Mentioned in older transcripts/spec as a decode-free projection; superseded by the `GZIPPY_BYPASS_*` family above. Not a live read-site today — kept here so a transcript reference resolves. |
| `GZIPPY_OVERLAP_WRITER` | output_writer.rs:34 | behavior | Background overlap writer ON; OFF==the inline writev. A/B of the writer structure. |
| `GZIPPY_DISABLE_WRITEV` | src/decompress/io.rs:17 | behavior | Disable the writev output path (fall back to plain writes). |
| `GZIPPY_SKIP… ` see SKIP_WRITEV_SYSCALL above | | | |
| `GZIPPY_MMAP_OUTPUT` | io.rs:172 | behavior | mmap-direct decode output (§3.13). |
| `GZIPPY_WRITEV_CAP_KIB` | fd_vectored_write.rs:135 | behavior | Cap the writev batch size in KiB; unset==identity. |
| `GZIPPY_CHUNK_KIB` | single_member.rs:163 | behavior | Override chunk granularity (vendor parity probe). |
| `GZIPPY_BURST_PREFETCH` | chunk_fetcher.rs:533 | behavior | Lever on the prefetch saturation arg. |
| `GZIPPY_NO_PREFETCH` | chunk_fetcher.rs:1230 | behavior | Disable prefetch (also a known-clean A/B vs a T≥2 corruption case). |
| `GZIPPY_PREFETCH_CACHE_CAP` | chunk_fetcher.rs:547 | behavior | Cap the prefetch cache entries (=N); production untouched, OFF==identity. |
| `GZIPPY_NO_PUBLISH_AHEAD` | chunk_fetcher.rs:2714 | behavior | Disable publish-ahead overlap. |
| `GZIPPY_EAGER_POSTPROC` | chunk_fetcher.rs:1135 | behavior | Optional duplicate-probe full-cache scan (=1). |
| `GZIPPY_DRAIN_LONE` | chunk_fetcher.rs:3703 | behavior | Lone-chunk drain variant (manual bisect A/B; byte-exact check). |
| `GZIPPY_MARKER_RING` | gzip_chunk.rs:969 / marker_inflate.rs:273 | behavior | Restore the legacy marker-ring path for A/B only. |
| `GZIPPY_U16_CLEAN_TAIL` | lut_bulk_inflate.rs:747 | behavior | =1 uses the reinterpreted-window clean-tail fast path (vendor's). |
| `GZIPPY_FOLD_NODRAIN` | gzip_chunk.rs:151 | behavior | ContigFoldSink skips the drain step (A/B). |
| `GZIPPY_FOLD_NOCRC` | gzip_chunk.rs:162 | behavior | Skip per-clean-byte CRC update (A/B; produces non-verifying output by design). |
| `GZIPPY_STORED_INLINE_COPY` | stored_split.rs:635 | behavior | Force the inline stored-copy path at any T. |
| `GZIPPY_STORED_NO_OVERLAP` | stored_split.rs:397 | behavior | Force the old sequential stored order (prefix then …). |
| `GZIPPY_PACKED_LIT_STORE` | resumable.rs:1213 | behavior | Use the packed-literal multi-store path in the inner loop. |
| `GZIPPY_POISON_RESERVE` | segmented_buffer.rs:41 | test | TEST-ONLY reserved-tail poison (opt-in) to catch over-reads. |
| `GZIPPY_HUGEPAGE` | chunk_buffer_pool.rs:120 | behavior | Request hugepages for the chunk buffer pool. |
| `GZIPPY_MANUAL_BUFFER_POOL` | chunk_buffer_pool.rs:214 | behavior | Restore the legacy mutex buffer pool for A/B. |
| `GZIPPY_SHARED_POOL` | chunk_buffer_pool.rs:175 (comment only) | dead | NOT a live knob — the shared-pool experiment was removed 2026-05-28 and only a comment naming it remains; no `env::var` read exists. (Listed so a transcript reference resolves.) |
| `GZIPPY_SLAB_ALLOC` | rpmalloc_alloc.rs:22 | behavior | Route huge allocations through a slab allocator. |
| `GZIPPY_SLAB_BUDGET_MIB` | rpmalloc_alloc.rs:210 | behavior | Cap the slab allocator total budget in MiB (the real "slab cap"; there is no `GZIPPY_SLAB_CAP`). |
| `GZIPPY_SLAB_THRESHOLD_KIB` | rpmalloc_alloc.rs:60 | behavior | Minimum allocation size (KiB) routed to the slab vs the default allocator. |
| `GZIPPY_SLAB_MAX_T` | rpmalloc_alloc.rs:124 | behavior | Cap the thread count above which the slab allocator engages. |
| `GZIPPY_SLAB_TRACE` | rpmalloc_alloc.rs:263 | instrument | `=1` one stderr line per slab event (hit/miss/evict). OFF==identity. |
| `GZIPPY_STAGING_POOL_CAP` | staged_bits.rs:49 | behavior | Cap (=0 disables) the staged-bits pool. |
| `GZIPPY_MEM_BALLAST_MIB` | mem_stats.rs:21 | perturbation | Positive control: each worker allocates N MiB ballast (memory-pressure perturbation). |
| `GZIPPY_BODY_FAIL_LOG` | gzip_chunk.rs:380 | instrument | Per-failure structured JSON log of body-decode failures. |
| `GZIPPY_REGEN_FIXTURES` | zopfli_pure/tests.rs:10 | test | Regenerate the zopfli_pure test fixtures. |
| `GZIPPY_FUZZ_ITERS` | inflate_fuzz_loop.rs:14 | test | Inflate fuzz-loop iteration budget. |
| `GZIPPY_FUZZ_SEED` | inflate_fuzz_loop.rs:12 | test | Inflate fuzz-loop RNG seed (determinism). |
| `GZIPPY_BUILD_FEATURES` | (guest drivers) | build | Cargo feature set the guest build scripts pass (pure-rust-inflate / gzippy-native / gzippy-isal). |
| `GZIPPY_SHA` | (guest drivers) | build | Records the built commit sha for provenance in driver output. |

### Index — engine / marker / asm knobs (added 2026-06-12 source-truth pass; were live but undocumented)

| Knob | Read-site | CLASS | What it does |
|------|-----------|-------|--------------|
| `GZIPPY_ASM_KERNEL` | asm_kernel.rs:279 | behavior | Kill-switch for the BMI2 inline-asm contig kernel: `=0` forces the Rust loop (production is ON when BMI2 is present). The active asm campaign's production toggle. |
| `GZIPPY_ASM_STATS` | asm_kernel.rs:287 | instrument | `=1` enables the asm kernel's effect-verification counters (proves the asm actually executed). OFF==identity. Supports the active asm campaign. |
| `GZIPPY_CONTIG_PROF` | contig_prof.rs:63 | instrument | `=1` profiles the contiguous-fold (clean-bulk) path. OFF==identity. |
| `GZIPPY_DIST_AMORT` | marker_inflate.rs:3824 | behavior | `=0` disables distance-decode amortization in marker decode (production default ON). |
| `GZIPPY_EXACT_BLOCK` | gzip_chunk.rs:289 | behavior | **Default ON** (`map_or(true,…)`); `=0` disables the exact-block-boundary chunk decode. A production-default knob, not OFF==identity. |
| `GZIPPY_SEEDED_BLOCK` | gzip_chunk.rs:261 | behavior | **Default ON**; `=0` disables seeded-block decode. A production-default knob, not OFF==identity. |
| `GZIPPY_ISAL_INCREMENTAL_GROWTH` | gzip_chunk.rs:207 | behavior | `=1` enables incremental ISA-L output-buffer growth (x86/Linux ISA-L oracle path only). |
| `GZIPPY_ISAL_INITIAL_FACTOR` | gzip_chunk.rs:210 | behavior | Initial ISA-L output-buffer size factor (x86/Linux ISA-L oracle path). |
| `GZIPPY_ISAL_GROW_MIB` | gzip_chunk.rs:215 | behavior | ISA-L output-buffer growth step in MiB (x86/Linux ISA-L oracle path). |
| `GZIPPY_MARKER_DIST_STATS` | marker_inflate.rs:366 | instrument | `=1` marker distance-decode statistics. OFF==identity. |
| `GZIPPY_MARKER_DIST_TABLE` | marker_inflate.rs:482 | behavior | `=0` disables the marker distance LUT (production default ON). |
| `GZIPPY_MFAST_DISABLE` | slow_knob.rs:348 | behavior | **(keep)** Kill-switch: disables the marker fast-path (`mfast`) — a documented same-binary A/B arm. |
| `GZIPPY_MFAST_PROF` | marker_inflate.rs:411 | instrument | `=1` profiles the marker fast-path. OFF==identity. |
| `GZIPPY_NO_MFAST_LOCALBITS` | marker_inflate.rs:499 | behavior | **(keep)** Kill-switch (`=1`): use the pre-change struct-field bit path in the mfast loop — the F-w1 causal A/B arm. In a protected decode-loop file. |
| `GZIPPY_NO_HIT_DRIVE` | chunk_fetcher.rs:1202 | behavior | **(keep)** Kill-switch (`=1`): disable the prefetch hit-drive. |
| `GZIPPY_NO_STORED_FLIP` | marker_inflate.rs:311 | behavior | `=1` disables the stored-block u16→u8 flip (A/B). |
| `GZIPPY_NO_STOREDPAR_DEMOTE` | stored_split.rs:62 | behavior | **(keep)** Kill-switch: when unset, the stored path may demote to sequential; set to suppress that demotion. |
| `GZIPPY_NO_REFILL_STAGING` | isal_decompress.rs:960 | behavior | `=1` disables refill staging in the ISA-L streaming path (x86/Linux only). |
| `GZIPPY_WINDOW_SPARSITY` | sm_driver.rs:33 | behavior | **(keep)** `=1` enables window-sparsity handling on the SM driver. |
| `GZIPPY_SLOW_MFAST_MODE` | slow_knob.rs:364 | perturbation | **(keep)** Marker-fast-path slow factor — a production slow_knob marker variant (rule 1; pair with a SLEEP control). |
| `GZIPPY_PREFAULT_ARENA` | chunk_buffer_pool.rs:219 | behavior | Prefault the chunk-arena pages up front (page-warmth A/B). |
| `GZIPPY_ORACLE_RECORD` | removal_oracle.rs:125 | oracle | RECORD pass for the removal-oracle family: capture per-chunk results to a file. |
| `GZIPPY_ORACLE_NOSTORE` | removal_oracle.rs:79 | oracle | Replay with the literal-store/back-ref-copy term REMOVED → store-removed ceiling. |
| `GZIPPY_ORACLE_NODECODE` | removal_oracle.rs:145 | oracle | Replay with the Huffman-decode term REMOVED → decode-removed ceiling. |

---

## Regenerate / audit

To confirm this table still covers the source:

```sh
rg -o 'GZIPPY_[A-Z_0-9]+' src/ | sed 's/.*\(GZIPPY_[A-Z_0-9]*\)/\1/' | sort -u
```

Any name in that list NOT in the table above is a new, undocumented knob — add it
with its read-site (`rg -n NAME src/ | head -1`) and CLASS before using it in a
turn. (A couple of entries above are family prefixes — `GZIPPY_BYPASS_` etc. — that
cover several concrete `GZIPPY_BYPASS_*` knobs already itemized.)

## Deletable-candidates (2026-06-12 source-truth audit — for the supervisor to action)

A knob is DELETABLE iff (no banked cite in `orchestrator-status.md` or `plans/archive/`)
AND (no live test) AND (it is not a documented kill-switch/instrument supporting active
machinery). The audit (`scripts`-free; `grep` callers + cites + tests) found these
no-banked-cite + no-test knobs. Each is left in place pending supervisor judgment because
all are entangled with active machinery, woven through active code paths at several sites,
or sit in protected decode-loop files:

- **Standalone abandoned probes (safest to delete, but each gates a helper/closure):**
  `GZIPPY_STORED_PHASE_TIMING` (stored_split `time_phase` wrapper, 9 closure call-sites),
  `GZIPPY_BODY_FAIL_LOG` (gzip_chunk debug-log helper),
  `GZIPPY_SLAB_TRACE` (rpmalloc, 6 call-sites in the *active* slab path),
  `GZIPPY_STORED_INLINE_COPY` / `GZIPPY_STORED_NO_OVERLAP` (stored A/B branches),
  `GZIPPY_MMAP_OUTPUT` / `GZIPPY_NO_PUBLISH_AHEAD` / `GZIPPY_DISABLE_WRITEV` (output A/B),
  `GZIPPY_HUGEPAGE` / `GZIPPY_BURST_PREFETCH` / `GZIPPY_PREFETCH_CACHE_CAP` / `GZIPPY_EAGER_POSTPROC` (fetcher behavior A/B),
  `GZIPPY_FOLD_NOCRC` (fold A/B).
- **KEEP despite no direct cite (support active machinery / are kill-switches):**
  `GZIPPY_ASM_STATS` (effect-verification for the active asm campaign),
  `GZIPPY_PID` / `GZIPPY_TRACE_DETAIL` (sub-knobs of the active TIMELINE/TRACE instrument),
  `GZIPPY_NO_MFAST_LOCALBITS` (kill-switch A/B in a protected decode-loop file),
  `GZIPPY_RPMALLOC_STATS` (cited in the active fulcrum2 charter),
  the `GZIPPY_BYPASS_*` and `GZIPPY_ORACLE_{RECORD,NOSTORE,NODECODE}` families (coherent
  oracle machinery; deleting one member breaks the family),
  `GZIPPY_NO_REFILL_STAGING` / `GZIPPY_PACKED_LIT_STORE` (ISA-L / inner-loop, defer).

Removing any of the first group is behavior-preserving for production (the gated path is
dead when the env is unset and the env is never set in prod), but each touches active code
and must be re-verified on Linux (gzippy-isal / default features) as well as pure-rust.
