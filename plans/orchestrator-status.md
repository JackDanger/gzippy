# Orchestrator status — NAMING TRUTH + TWO-PATH + 3-WAY FULCRUM mission

## NEW MISSION (2026-06-06, user-set) — SUPERSEDES all prior arcs
Prior "full ISA-L port as a 2nd engine" plan is DEAD (critically reviewed: redundant
2nd inflate vs ONE-engine memory; x86-only strands arm64). This mission, in order:

- PHASE 1 — NAMING TRUTH PASS (behavior-neutral renames). Every function/type/module/
  dataset/span whose name misdescribes behavior gets renamed; all call sites, doc
  comments, trace/span labels updated. FLAGSHIP: IsalInflateWrapper on the pure-Rust
  build does NOT wrap ISA-L — it delegates to pure-Rust Inflate. Audit whole
  src/decompress/parallel/ + src/decompress/inflate/. Gate: build + byte-exact tests +
  silesia sha 028bd002c89c9a909ccdbc2af0a223de285348edb014ccc8e27d297f52cb410f.
  Commit: "rename(parallel/sm): naming truth — names describe behavior".
- PHASE 2 — TWO FLAG-GATED PATHS + DELETE dead.
  - gzippy-isal feature: FAITHFUL rapidgzip — marker engine + 32KiB clean-handoff to
    REAL ISA-L via FFI (vendor GzipChunk.hpp:520-525). C-FFI behind THIS flag ONLY.
  - gzippy-native feature (DEFAULT prod): UNIFIED ONE-ENGINE marker decode, single pass,
    pure Rust, flip-in-place clean tail, NO 2nd engine, NO FFI, ISA-L hot techniques
    grafted (BMI2 PEXT/BZHI dispatch OFF unified.rs:122; multi-symbol LUT; lean refill).
  - Each path BYTE-EXACT. native also diff-tested vs gzippy-isal oracle + flate2.
    DELETE flagged-off unused impls. Advisor-gate each deletion.
- PHASE 3 — FULCRUM 3-WAY: locked harness (neurotic, T8 silesia, interleaved best-of-N,
  sha-verified) for rapidgzip vs gzippy-isal vs gzippy-native. Report 3-way wall + d_c/d_w.

## CONSTRAINTS
OOM: serialize ALL builds (pgrep first; ONE globally; WAIT). Prefer neurotic. Never two
building subagents. No duplicate orchestrators. Long ops detached + sentinel, polled.
Byte-exact ABSOLUTE. Delegate ALL hands-on work; advisor-corroborate consequential claims.
Numbers ONLY from full locked harness. Clean base HEAD 412b3ac. KEEP #B + byte-exact.

## PHASE 1 — IN PROGRESS
- Worktree: .claude/worktrees/phase1-naming-truth (branch phase1-naming-truth from 412b3ac).
- Observed: inflate_wrapper.rs has TWO IsalInflateWrapper structs (line 154 real-ISA-L cfg,
  line 340 pure-Rust cfg). 9 src files reference IsalInflateWrapper.
- STEP 1 [running]: design subagent produces rename MAP (old->new + rationale). Advisor-
  review. Then impl subagent applies. Then gate.

### STEP 1 DONE — rename MAP produced + advisor-reviewed (APPROVE WITH CHANGES)
- Governing fact (advisor-confirmed via build.rs:81-83): pure_inflate_decode==parallel_sm;
  the old ISA-L decode arm was REMOVED; NO live real-ISA-L decode in parallel/+inflate/.
  Every isal-named symbol there is FALSE-isal (pure-Rust). TRUE-isal lives only in
  src/backends/ + `feature=isal-compression` GATES (not names) — DO NOT rename.
- Approved renames: IsalInflateWrapper->StreamingInflateWrapper (both arms, ~28 code +
  exclude ~12 vendor doc refs); isal_huffman_pure.rs->lut_huffman.rs (mod.rs:46);
  IsalLitLenCodePure->LutLitLenCode (10); IsalDistCodePure->LutDistCode (6);
  isal_lut_bulk.rs->lut_bulk_inflate.rs (mod.rs:48, 19 path refs);
  isal_litlen_pure->lut_litlen (4); isal_lut_litlen_rebuild->lut_litlen_rebuild (3);
  isal_lut_litlen_decode->lut_litlen_decode (2).
- DEFER: trace span worker.isal_stream_inflate (cross-coupled to patch_vendor.sh; renaming
  one side breaks `fulcrum vs` — not behavior-neutral for tooling). KEEP ISAL_* vendor
  constants (traceability). 
- ADD (advisor): also rename DecodePath::IsalParallelSM/IsalSingle in src/decompress/mod.rs
  (flagship FALSE-isal; the GZIPPY_DEBUG path= contract) -> ParallelSM/SingleMember.
  Update GZIPPY_DEBUG label + routing tests + CLAUDE.md mentions consistently.
- STEP 2 [DONE 2026-06-06]: rename applied + gated + COMMITTED.
  - Commit 6b7d6f5 on branch phase1-naming-truth (worktree .claude/worktrees/phase1-naming-truth),
    parent 412b3ac. Diff symmetric pure-rename + a cargo-fmt pass (hook-required: import
    reorder/line-wrap of the renamed code; still behavior-neutral).
  - Gate (deadlock-safe — SINGLE serialized cargo runs, timeout-wrapped, --test-threads=1,
    load-flaky tests skipped):
      * build clean (release, --features pure-rust-inflate); cargo fmt --check clean.
      * lib tests 35 passed / 0 failed (routing:: + pure_rust_inflate_corpus::; skipped
        not_slower/diff_ratio/scoped_cancel/hot_path/alloc_budget).
      * silesia sha256 028bd002...410f == reference, emitted via path=ParallelSM
        (GZIPPY_DEBUG confirms the renamed contract; was IsalParallelSM).
  - NOTE: env lacks a subagent-spawn tool (only TaskStop surfaced); leader executed the
    mechanical gate directly. Phase 2 design/impl/profiling will need the spawn tool —
    flag to supervisor if delegation is required there.

## PHASE 1 — DONE. PHASE 2 — IN PROGRESS.

### STEP 0 [DONE 2026-06-06] — rename integrated onto working branch
- `git merge --ff-only phase1-naming-truth` -> reimplement-isa-l now at 6b7d6f5 (linear ff).
- VERIFIED: StreamingInflateWrapper present (7+ src files); 8 remaining IsalInflateWrapper
  refs are ALL intentional vendor doc-citations (rapidgzip::IsalInflateWrapper / gzip/isal.hpp
  in backends/isal_decompress.rs, inflate/resumable.rs, inflate/staged_bits.rs,
  parallel/inflate_wrapper.rs) — kept for traceability per approved MAP. Renamed files
  lut_bulk_inflate.rs + lut_huffman.rs present; old isal_* names gone.
- Build clean: release --no-default-features --features pure-rust-inflate, 34s, warnings only.
- pgrep NOTE: bare `pgrep -f 'cargo|rustc'` FALSE-POSITIVES on Cursor helpers/shells whose
  cwd contains "isal". Use `pgrep -x cargo; pgrep -x rustc` for the build-serialization gate.

### STEP 1 [DESIGN DONE — advisor-review pending] — two flag-gated paths
DESIGN SUBAGENT (exit 0, ~325s) findings + leader-corroborated:
- KEY: the unified pure-Rust path ALREADY EXISTS and is the SOLE decode path. NO live
  ISA-L decode anywhere in parallel/+inflate/ (build.rs:80-82 pure_inflate_decode==parallel_sm).
  => gzippy-native = today's pure path made default (STEP 1 trivial wiring). gzippy-isal =
  RE-INTRODUCE a real-ISA-L clean tail behind a flag (the path needing new code).
- CAVEAT (leader-verified): gzippy-isal FFI building block EXISTS and is tested:
  isal_decompress.rs:307 decompress_deflate_from_bit (+_with_end:460, +_with_boundaries:643)
  — bit-offset + 32KB-dict tests pass.
- CORRECTION (advisor caught my error): the stopping-point patch IS present — in the C
  SUBMODULE vendor/isa-l/igzip/igzip_inflate.c:1412,1747 + header igzip_lib.h:232 ("gzippy/
  rapidgzip patch") + bindings isal-sys/src/igzip_lib.rs:1565-1569 (full END_OF_BLOCK[_HEADER]/
  END_OF_STREAM[_HEADER] enum). My earlier "no patch" grep checked the Rust wrapper, not the C
  submodule — WRONG. So a faithful tail with REAL block boundaries is available.
- Today's chunk decode: marker phase (marker_inflate::Block, 128KiB u16 ring) -> FlipToClean at
  clean_appended_len>=MAX_WINDOW_SIZE (gzip_chunk.rs:1191, mirrors GzipChunk.hpp:521) ->
  finish_decode_chunk_impl (gzip_chunk.rs:354) constructs StreamingInflateWrapper (pure-Rust
  Inflate<Clean,Generic,Streaming> -> ResumableInflate2, a SECOND 128KiB-staging engine).
- CACHE MANDATE GAP: "one engine, flip-in-place" NOT yet met — TWO engines/buffers per chunk
  (~288KiB scratch). That fold is the substantive native work (sequenced last, A/B-gated).
- WIRING PLAN: gzippy-native=["pure-rust-inflate"]; gzippy-isal=["pure-rust-inflate",
  "isal-compression"]; new build.rs cfg isal_clean_tail = is_x86_64 && CARGO_FEATURE_GZIPPY_ISAL.
  KEEP default=[] through STEP 1+2 (flipping surfaces parallel-module lint debt per Cargo.toml:46-48);
  flip to gzippy-native after STEP 2 clears debt. arm64: gzippy-isal degrades to pure tail (==native).
- SEQUENCE: 1a wiring(zero-behavior, native==today byte-identical) -> 1b isal tail (x86_64) ->
  STEP2 delete dead (route_c_*, not-parallel_sm stub CONFIRMED; consume_first_decode big FASTLOOP
  fns + jit_decode/specialized_decode/double_literal + MarkerRing NEEDS-DEEPER-CHECK) ->
  native cache work (share fixed tables / pool engines / fold-to-one-engine / BMI2 runtime dispatch).

### ADVISOR VERDICT (exit 0, ~275s): APPROVE-WITH-CHANGES. Strategy + native sound. 4 fixes
incorporated into the plan (all leader-verified):
- FIX1 native drops ISA-L *compression* backend: release/CI ship --features isal-compression
  (release.yml:132) giving x86 T1 L0-L3 ISA-L COMPRESS; gzippy-native=["pure-rust-inflate"]
  loses it. FFI-free mandate is DECODE-ONLY. DECISION: gzippy-native keeps isal-compression for
  COMPRESS on x86_64 only (decode stays pure-Rust/FFI-free). Update Makefile + .github/workflows
  to new feature names so the gated path == what ships.
- FIX2 arm64 C-build: isal-sys/build.rs has NO arch guard (verified: handles riscv64/win/macos,
  no x86 gate) -> gzippy-isal pulling isal-compression would try to build ISA-L C on arm64. Add
  build.rs compile_error! when gzippy-isal && !x86_64. gzippy-isal = x86_64-only ref baseline.
- FIX3 add CI line for the new combo pure-rust-inflate+isal-compression (not built today).
- FIX4 (BIGGEST RISK) the isal tail is NOT at parity just because _with_boundaries records
  boundaries. Divergences: pure tail coalesce-stops at first boundary AT-OR-PAST stop_hint_bits
  (gzip_chunk.rs:387) & commits CRC only up-to-boundary bytes (:507-516); isal fns run to
  BFINAL over whole slice. Stopping-point set differs (isal _with_boundaries requests only
  END_OF_BLOCK :666 vs pure END_OF_BLOCK|_HEADER|STREAM_HEADER :382-384). Output: pure STREAMS
  into writable_tail segment-by-segment; isal returns one Vec (needs append+offset glue,
  mirror :428). Two stop modes until_exact true/false (:374). VERIFY-FIRST GATE for 1b: a
  same-input pure-vs-isal differential on real silesia chunks asserting IDENTICAL (decoded
  bytes, committed length, end_bit handoff, per-chunk CRC) before any production wiring.
- Keep default=[] through STEP1+2; silesia gate ALWAYS runs explicit --features gzippy-native
  (bare cargo compiles no decode path: routing tests are #[cfg(parallel_sm)]).

### STEP 1a [DONE 2026-06-06] — zero-behavior wiring
- Cargo.toml: gzippy-native=["pure-rust-inflate"]; gzippy-isal=["pure-rust-inflate",
  "isal-compression"]. build.rs: cfg isal_clean_tail = is_x86_64 && CARGO_FEATURE_GZIPPY_ISAL
  && parallel_sm; panic!() if gzippy-isal && !x86_64 (arm64 C-build guard, FIX2).
- GATE (--features gzippy-native): build clean 34s (same 6 pre-existing warnings as
  pure-rust-inflate => pure alias). silesia sha 028bd002...410f == reference via path=ParallelSM.
  lib tests 35 passed / 0 failed (routing:: + pure_rust_inflate_corpus::, --test-threads=1,
  load-flaky skipped). Native is byte-identical to today's decode, by construction.
- No isal_clean_tail code arm yet (that's 1b). NEXT: 1b verify-first differential gate then tail.

### STEP 1b [IN PROGRESS — impl subagent running, builds x86_64 via Rosetta]
- Rosetta prereqs VERIFIED: target x86_64-apple-darwin installed; arch -x86_64 uname -m == x86_64.
  gzippy-isal builds ONLY on x86_64 (build.rs panics on arm64) so Rosetta is mandatory for it.
- Subagent task: (DELIVERABLE 1, the GATE) a #[cfg(isal_clean_tail)] differential test on REAL
  silesia chunks asserting pure-tail vs isal-tail parity on (decoded bytes, committed length,
  final_bit handoff, per-chunk CRC, block boundaries) for BOTH until_exact true/false; report
  exact divergence mechanism if any. (DELIVERABLE 2, only if parity achievable) the
  #[cfg(isal_clean_tail)] arm in finish_decode_chunk_impl. Build-serialized (pgrep -x), no commit.
- IMPORTANT: leader must NOT start any build while this subagent runs (it owns the build lock).

### STEP 1b [GATE DONE + verdict — committed 8d026a8] — faithful isal tail DEFERRED to last
GATE RESULT: differential PASSES 10/10 (5 silesia chunks x until_exact{T,F}) — ISA-L tail
records byte-identical (committed bytes, length, final_bit, CRC, boundaries) vs pure tail.
Existence proof a faithful gzippy-isal tail is achievable. Test-only, cfg(all(test,isal_clean_tail)).
NOTE: arch -x86_64 cargo IMPOSSIBLE (cargo is arm64-only binary); the working x86_64 path is
NATIVE cargo cross-compile --target x86_64-apple-darwin with CARGO_BUILD_RUSTFLAGS="-C
target-cpu=x86-64" (override .cargo/config.toml target-cpu=native), runs under Rosetta.

ADVISOR VERDICT (2nd advisor, independent rewind-logic verification): 
- Design (A) CORRECT (swap inner engine for interface-compatible ISA-L wrapper, driver loop
  unchanged, accounting matches by construction incl. the :478-489 rewind). Design (B)
  decode-to-end+reselect is UNFAITHFUL by construction (handoff bit is driver-loop-history-
  dependent: :486 rewinds to last_eob_pos recorded at :475 on the PRIOR EOB). No thinner shim.
- SEQUENCING (the real call): STEP 2 (delete dead) -> gzippy-native cache work -> gzippy-isal
  Design-A tail LAST, immediately before PHASE 3. WHY: isal is an INSTRUMENT not a deliverable;
  native's fold-to-one-engine WILL REWRITE the driver interface Design-A swaps into (build isal
  now => re-port; build last => one port); gate 8d026a8 already proves pure≡isal so native
  doesn't need isal as oracle (native gated by flate2+libdeflate+canonical sha, arm64-portable);
  defer the riskiest C-FFI re-add until PHASE 3 actually consumes it.
- CONSTRAINT: STEP 2 must NOT delete finish_decode_chunk_impl (live prod + Design-A insertion
  point). native fold should cfg-FORK it (#[cfg(isal_clean_tail)] keeps two-phase; native folded),
  NOT delete. Delete only the genuinely-dead alts (route_c_*, jit_decode, specialized_decode,
  double_literal, not-parallel_sm stub) — advisor-gate each.
- BEFORE any Design-A impl: harden the gate to exercise the :486 rewind (a non-final/non-fixed
  rewind case; a fixed-Huffman no-rewind case per :481 not_fixed; the ==stop_hint case) AND
  verify ISA-L's bit_position at END_OF_BLOCK[_HEADER] matches the pure tell_compressed()
  convention. Q3+Q4 are the SAME gate.

## STEP 2 — DELETE DEAD INFLATE IMPLS [IN PROGRESS 2026-06-06]
### Deletion MANIFEST produced (read-only subagent, exit 0) — far more conservative than expected
- KEY FINDING: a SECOND live decode path (scan_inflate.rs -> decompress::index -> index_mode.rs ->
  CLI --index/--seek, cli.rs:212) compiles under BOTH features and keeps jit_decode,
  specialized_decode, double_literal, bmi2, libdeflate_entry, consume_first_decode ALL LIVE.
  bgzf.rs inflate_consume_first calls are #[cfg(test)]-only (not a live anchor).
- NO dead parallel/ module found — every #[cfg(parallel_sm)] module has a real chunk-pipeline
  consumer (decode_bypass LIVE @ sm_driver:64/chunk_fetcher; marker engine LIVE; etc).
- Only TRULY dead: ONLY route_c_dynasm.rs + route_c_fixed.rs (gated on feature route-c-dynasm
  NOT pulled by default/pure-rust-inflate/gzippy-native/gzippy-isal/tests; only mod.rs:25-28 refs).
- PARTIAL: libdeflate_decode.rs — KEEP get_fixed_tables (live @ resumable.rs:951); dead big fns
  (decode_libdeflate, inflate_libdeflate, copy_match, file-local BitReader, decode_with_double_cache
  + their in-file #[cfg(test)] tests 979-1196) have ZERO external callers.
- PROPOSED BATCHES: B1 route_c_* (zero prod/test impact); B2 dead libdeflate_decode fns;
  B3 (deferred, needs decode_dynamic dispatch trace) consume_first_decode internal prune +
  double_literal reclassification. MUST-PRESERVE verified: finish_decode_chunk_impl + STEP-1b gate.
- ADVISOR REVIEW [DONE, exit 0]: APPROVE Batch1; APPROVE-WITH-CHANGES Batch2 (manifest had 2
  WRONG symbol names: BitReader->LibdeflateBits; decode_with_double_cache->decode_huffman_double_lit;
  advisor enumerated the real dead cluster by reading the file). Both riskiest claims independently
  grep-confirmed. Gate adds: include plain cargo build/test; Batch2 gate must exercise --index/scan
  path (tests::index). Retire route-c-dynasm feature+deps+sub-crate = SEPARATE follow-up (not byte-neutral).

### BATCH 1 [DONE — committed 03f592e] delete route_c_dynasm.rs + route_c_fixed.rs + mod.rs:25-28
- DUAL-SHA gate GREEN: gzippy-native + gzippy-isal(Rosetta x86_64 cross-compile, --target
  x86_64-apple-darwin, CARGO_BUILD_RUSTFLAGS=-C target-cpu=x86-64) BOTH emit silesia sha
  028bd002...410f via path=ParallelSM. routing 30/0, correctness 130/0, pure_rust_inflate_corpus
  5/0 (incl silesia), index 22/0; plain cargo build clean. NOTE: native cargo build --features
  gzippy-isal on arm64 PANICS by design (build.rs:83 guard) — MUST use Rosetta cross-compile.

### BATCH 2 [DONE — committed edcd863] prune dead big-decoder cluster in libdeflate_decode.rs
- Replaced file with ONLY get_fixed_tables + FIXED_TABLES (was ~1200 lines). Removed the dead
  cluster (zero external callers under either feature). KEEP get_fixed_tables (live @
  resumable.rs:951 + consume_first_decode.rs + double_literal.rs).
- RESUMED after restart: prior leader's interrupted edit reconciled under the NEW global build
  mutex scripts/cargo-lock.sh (portable mkdir-lock; macOS has no flock). DUAL-SHA GREEN:
  gzippy-native (arm64) + gzippy-isal (Rosetta x86_64 cross-compile) both emit silesia
  028bd002...cb410f via path=ParallelSM. native suites 187/0 (routing/correctness/
  pure_rust_inflate_corpus/index, --test-threads=1, load-flaky skipped). clippy clean (fixed
  inherited needless_range_loop in get_fixed_tables, behavior-identical).
- STRUCTURAL FIX: ALL cargo/rustc/test now wrapped `scripts/cargo-lock.sh <cmd>` — concurrent
  builds structurally impossible. Every subagent MUST use it.

### BATCH 3 [DONE — committed 70ec5ff] dead double-literal + short-bits caches
- AFTER restart: read-only scout did the decode_dynamic dispatch trace (the thing that made B3
  ambiguous) + an independent advisor read corroborated. BOTH confirm decode_huffman_cf:1505 is
  LIVE (fixed-block + dynamic-block paths) and KEPT. Everything deleted has ZERO callers under
  gzippy-native, gzippy-isal, OR scan_inflate/index, plus tests/benches/examples.
- Deleted: double_literal.rs, huffman_short_bits_cached_deflate.rs, huffman_short_bits_multi_cached.rs
  (whole files + mod decls); 9 dead fns in consume_first_decode.rs (3580->2628 lines); stale
  doc-comment refs fixed. NOT touched: finish_decode_chunk_impl; specialized_decode cluster
  (production-dead but SpecializedCache coupled to SPEC_CACHE + test stats — coupled edit, deferred).
- DUAL-SHA gate GREEN (under scripts/cargo-lock.sh): native + isal(Rosetta x86_64) both
  028bd002...cb410f via path=ParallelSM; native suites 187/0; clippy clean; plain build clean.

## STEP 2 — DONE (B1 route_c 03f592e, B2 libdeflate_decode edcd863, B3 70ec5ff).
The ONLY remaining production-dead cluster (specialized_decode + SPEC_CACHE stats) is a coupled
edit, intentionally deferred — not a clean leaf delete.

## NATIVE CACHE FOLD — design + advisor DONE 2026-06-06; impl in progress
- DESIGN (subagent, read-only) + ADVISOR review (both this session). Two engines per chunk confirmed:
  Engine M = marker_inflate::Block (128KiB u16 ring; faithful port of vendor deflate.hpp deflate::Block,
  has BOTH markers + clean drain @ marker_inflate.rs:738-750). Engine C = StreamingInflateWrapper ->
  ResumableInflate2 (128KiB staging staged_bits.rs:22 + 32KiB window resumable.rs:94/107 +
  last_32kib_window_vec gzip_chunk.rs:767 ~= 192KiB UNPOOLED Box churn/chunk), constructed @
  finish_decode_chunk_impl gzip_chunk.rs:354/379 after FlipToClean gzip_chunk.rs:1191. Engine C cites
  gzip/isal.hpp — it is the pure-Rust IsalInflateWrapper STAND-IN (belongs to gzippy-isal path).
- FIXED Huffman tables ALREADY SHARED (FIXED_TABLES OnceLock libdeflate_decode.rs:17). Dynamic tables
  inherently per-block (reuse, not share). So mandate's "share tables" already met for the shareable one.
- SURVIVOR FORK RATIFIED (advisor + governing memory): Engine M survives, Engine C deleted from native.
  Engine M == MarkerRing engine the governing memory mandates flip-in-place + continue clean tail on
  SAME cursor; Engine C (IsalInflateWrapper stand-in) stays on gzippy-isal -> real ISA-L FFI Design-A.
- ADVISOR CAUGHT A BUG in the naive first step: pooling the WHOLE Engine C is NOT byte-exact —
  set_window:336 reset is INCOMPLETE (misses encoded_until_bits, coalesce_stop_hint [concrete stale-hint
  divergence: set only when !until_exact gzip_chunk.rs:386-388], block_boundaries) AND a lifetime
  soundness problem (ResumableInflate2<'a> borrows per-chunk input, can't live in 'static thread_local).
- CORRECTED STEP 1 (impl now): pool the TWO HEAP BUFFERS only (128KiB staging Box + 32KiB window Box)
  via thread-local free-list; engine logic + lifetimes UNTOUCHED, only the allocator changes. Near
  byte-transparent. Gate: dual-sha + STEP-1b per-chunk differential. Then STEP 2 flip-in-place fold
  (Engine M continues clean tail, delete Engine C from native) + clean-drain bulk/BMI2 graft IN THE
  SAME step (avoid shipping a wall regression; cfg-fork finish_decode_chunk_impl so isal keeps two-phase).

### NATIVE CACHE STEP 1a [DONE — committed 736ea1a] pool 128KiB staging box
- Scoped to the STAGING box only (the larger, always-fully-overwritten one — provably byte-transparent:
  recycled box's bytes always overwritten by first reload_at_bit before any read). STAGING_POOL
  thread-local free-list (cap 4); buf is ManuallyDrop so Drop recycles instead of frees (no double free).
- DUAL-SHA GREEN: native + isal(Rosetta x86_64) both 028bd002...cb410f via path=ParallelSM. staged_bits
  unit tests 7/0 (reload/memcpy parity on pooled box); native suites 187/0; clippy clean.
- Hit + fixed a test-only E0507 (a unit test moved *staged.buf; now .to_vec()/.as_slice() through MD).
- STEP 1b TODO: 32KiB SlidingWindow pool — DEFERRED, needs explicit reset analysis (read-before-fully-
  written within a chunk, unlike staging). May fold into the STEP-2 flip instead (Engine C goes away).
- TODO: pin the ~ footprint numbers EXACTLY (mandate is measured, not asserted) — build an RSS-vs-T
  instrument before the STEP-2 behavioral fold (advisor: validate the instrument on this predictable
  per-chunk->per-thread delta first).

## (orig) NEXT: STEP 2 — delete dead inflate impls (advisor-gate each). DO NOT touch finish_decode_chunk_impl.
- Features today: default=[] ; pure-rust-inflate (prod decode, sets cfg parallel_sm==
  pure_inflate_decode in build.rs:81-83) ; isal-compression (TRUE ISA-L *compression* only,
  no decode) ; isal=[] (build.rs-set when static lib built). NO live ISA-L decode currently.
- New features to add: gzippy-isal (faithful: marker engine -> REAL ISA-L FFI decode handoff
  at 32KiB clean) and gzippy-native (DEFAULT prod: one unified pure-Rust engine, cache-resident
  per design mandate). Design subagent spawned to produce the two-column rapidgzip<->gzippy map
  + concrete wiring plan before any impl.

## RSS-vs-T INSTRUMENT [DESIGN 2026-06-06 — leader; advisor-corroborate then delegate impl]
Advisor-mandated prerequisite before the STEP-2 behavioral fold. Measures the cache mandate
(plans/gzippy-native-design-mandate.md): shared small hot working set, RSS ~flat as T rises.
RSS captured LOCALLY (arm64 gzippy-native); wall/MPKI stay on neurotic locked harness.

### What it must measure (per binary, per corpus)
1. PEAK total RSS vs T in {1,8,16}: /usr/bin/time -l "maximum resident set size" (bytes on macOS).
2. STEADY-STATE RSS time-series: background `ps -o rss= -p <pid>` poller @ ~50ms -> median of
   the plateau (excludes ramp/teardown) so per-thread growth is separable from one-shot allocs.
3. Derived PER-THREAD working-set slope: (RSS@T16 - RSS@T1)/15 and (RSS@T8-RSS@T1)/7; report both
   (non-linearity is signal). Mandate target: slope << per-thread L2; RSS roughly flat in T.
4. sha256 of decoded output EVERY run == 028bd002...cb410f (a low-RSS run with wrong bytes is void).

### Validation (instrument must earn trust BEFORE its numbers count — CLAUDE.md rule 4/PROCESS)
- POSITIVE control: a knob that PROVABLY inflates per-thread RSS must show up as increased slope
  (e.g. force a larger per-thread alloc, or compare a high-T run that we KNOW duplicates buffers).
- NEGATIVE/self control: same binary vs itself reads equal RSS within poller spread (define spread).
- POOLING DELTA (the real validation): build 70ec5ff (UNPOOLED staging) vs 736ea1a (POOLED staging),
  run both through the instrument at T8/T16. Hypothesis: pooled shows LOWER per-thread RSS slope
  (recycled staging box => fewer live 128KiB allocs at steady state). MEASURE — if peak RSS is
  unmoved (pool may only cut alloc *rate*, not peak footprint), SAY SO; that's a finding about what
  the instrument can/can't see, not a failure. Either way the +/- controls must pass for trust.

### Deliverable
scripts/bench/rss_vs_t.sh — args: BIN, GZ corpus, T-list, N trials; outputs the table above +
controls. ALL builds via scripts/cargo-lock.sh (global mutex). Per-T binary = same gzippy-native
release binary, T set by `-p <T>`. Report per-T peak+plateau RSS, slope, sha-OK, control verdicts.

### ADVISOR VERDICT on RSS instrument design: APPROVE-WITH-CHANGES (incorporated)
KEY REFRAME: process RSS is TOO COARSE for the mandate. Per-thread decode scratch at T16 ~=
128KiB*16 = 2MB against a ~300-400MB backdrop (68MB input mmap + ~211MB decoded silesia) = <1%,
below poller jitter. RSS-vs-T CANNOT resolve per-thread decode buffers. So:
- PRIMARY mandate instrument = DIRECT IN-PROCESS BYTE ACCOUNTING (zero noise floor), via GZIPPY_DEBUG
  counters: peak live bytes per thread-local pool, sum of shared read-only table bytes, count of
  distinct pool high-water boxes. Measures "tiny per-thread working set / shared tables / pooled
  buffers" EXACTLY. This is what honors the mandate.
- RSS-vs-T DEMOTED to a coarse T-scaling GUARD: catches only UNEXPECTED large T-scaling (e.g. a
  per-thread COPY of a shared table — large enough RSS would see it). Necessary-not-sufficient.
- cache-residency is NOT an RSS question -> MPKI (deferred to neurotic locked harness) is the real test.
FIXES incorporated:
- UNITS: /usr/bin/time -l max-RSS is BYTES on macOS; ps -o rss= is KiB. Normalize + assert (1024x trap).
- Stream output to /dev/null (bounded writer) so output buffering is T-invariant + small.
- time -l for PEAK only; ps plateau-median for live set; NEVER cross-compare the two as equal.
- POOLING DELTA = a FINDING, not a control (free-list may not move peak live set; null is ambiguous).
- POSITIVE CONTROL (must-pass, calibrated/signed/magnitude): env knob allocs+TOUCHES (writes every
  page) known N MiB per worker thread; instrument MUST recover slope ~= N at multiple N (linearity).
- NEGATIVE control: binary vs itself within poller spread (keep).
- T grid {1,2,4,8,16}, REGRESS slope (2-pt slope can't separate const overhead from per-thread growth);
  check linearity not endpoints. N>=7 interleaved, Delta<spread => TIE.
- Assert GZIPPY_DEBUG path=ParallelSM (native engine, not fallback) every run; sha gate every run.

### RSS instrument — REVISED deliverables (to delegate)
D1 (PRIMARY): in-process per-thread byte accounting behind GZIPPY_DEBUG (or a new env GZIPPY_MEM_STATS=1):
   instrument STAGING_POOL (+ later the 32KiB window pool / engine state) to record peak-live-bytes per
   pool + box high-water count; emit shared-table byte total (FIXED_TABLES etc). NEEDS a small src change
   (counters) — design subagent first maps WHERE to hook, advisor-gate, then impl. Byte-exact (counters only).
D2 (GUARD): scripts/bench/rss_vs_t.sh — peak (time -l, bytes) + plateau (ps, KiB) RSS over T{1,2,4,8,16},
   N>=7 interleaved, output->/dev/null, sha gate, native-path assert, slope regression + linearity.
D3 (CONTROL): GZIPPY_MEM_BALLAST_MIB=N env -> each worker allocs+touches N MiB; rss_vs_t must recover
   slope~=N at N in {8,16,32}. Pooling delta (70ec5ff vs 736ea1a) run + reported as a FINDING.

### RSS instrument — IMPL NOTE (impl subagent 2026-06-06) — exact hook lines + byte-transparency
New module `src/decompress/inflate/mem_stats.rs` (pub mod in inflate/mod.rs). COUNTERS ONLY, no decode
behavior change. `enabled()` reads `GZIPPY_MEM_STATS` ONCE into a OnceLock<bool>; every hook is
`if !enabled() { return; }` so the flag-off path is an inlined early-return (DUAL-SHA gate proves
byte-identical on/off).
HOOKS:
- staged_bits.rs take_staging_box (~35): split pop into reused vs Box::new arm, call
  `mem_stats::on_take(reused)`. on_take: relaxed-atomic alloc-vs-reuse counter; thread_local live++/peak;
  on a new per-thread peak records into a global Mutex<HashMap<ThreadId,ThreadStat>> (cold path).
- staged_bits.rs return_staging_box (~42): `mem_stats::on_return()` (thread_local live--). Drop path
  already calls return_staging_box (line 329) so per-chunk lifecycle is covered.
- staged_bits.rs STAGING_POOL_CAP -> env-overridable `pool_cap()` OnceLock (default 4); CAP=0 disables
  pooling. This is the chosen D3 POOLING-DELTA mechanism (env `GZIPPY_STAGING_POOL_CAP=0` vs 4 on the
  SAME 736ea1a binary) — isolates the pooling variable with one build, no worktree/cherry-pick drift.
  Default 4 == today's production behavior (byte- AND perf-identical when env unset).
- libdeflate_entry.rs: add `heap_bytes()` to LitLenTable + DistTable (size_of::<Self>() +
  capacity*size_of::<Entry>()) so report() prints the ONE shared-table footprint via get_fixed_tables().
- main.rs main() after `let result = run()`: `decompress::inflate::mem_stats::report();` BEFORE
  process::exit (which skips destructors). report() is itself enabled()-gated.
POSITIVE CONTROL: `GZIPPY_MEM_BALLAST_MIB=N` (own OnceLock switch, independent of MEM_STATS so the RSS
guard drives it without accounting overhead). ensure_ballast() on first take per thread allocs+touches
(4KiB stride) N MiB into a thread_local Vec held for thread life -> known per-thread resident slope.
Byte-transparent: ballast Vec is never read by decode.

## FLIP-IN-PLACE FOLD — leader orientation (gated on RSS instrument; STEP 2 next)
Read the two-engine boundary in gzip_chunk.rs + marker_inflate.rs:
- Engine M loop (run_marker_blocks-style, gzip_chunk.rs:1184-): on clean_appended_len>=MAX_WINDOW_SIZE
  && !ctx.flipped it returns MarkerStep::FlipToClean{end_bit_offset, window_len=32KiB} at :1191-1202,
  setting ctx.current_bit_offset=next_block_offset. Driver THEN constructs Engine C
  (StreamingInflateWrapper -> ResumableInflate2) in finish_decode_chunk_impl (:354) to decode the
  clean tail from that bit offset. THAT handoff is the two-engine cost the fold removes (native only).
- Engine M ALREADY CAN drain clean u8 in-place: marker_inflate.rs:731-758 drain_to_output ->
  push_clean_u8 (:750) once contains_marker_bytes==false (vendor deflate.hpp:1285-1292). So the fold =
  do NOT return FlipToClean in native; keep iterating Engine M's block loop on the SAME ctx cursor,
  letting drain_to_output emit clean bytes, until BFINAL/stop_hint. Engine C deleted from native;
  finish_decode_chunk_impl cfg-FORKED (#[cfg(isal_clean_tail)] keeps two-phase for gzippy-isal Design-A;
  native folded) — advisor CONSTRAINT: do NOT delete finish_decode_chunk_impl (Design-A insertion point).
- Gate: DUAL-SHA 028bd002...cb410f (native folded + isal two-phase), per-chunk differential (STEP-1b gate
  8d026a8 style), then measure wall+RSS on locked harness. Sequence the clean-drain bulk/BMI2 graft IN
  THE SAME behavioral step to avoid shipping a transient wall regression.

## LEADER RE-DRIVE 2026-06-06 (resumed @ 5f162bb) — singleton-leader discipline adopted
- SINGLETON-LEADER LOCK created scripts/leader-lock.sh (clone of cargo-lock mkdir-mutex + stale-pid
  reclaim; acquire/release/status verbs; NON-BLOCKING acquire => duplicate leader EXITS). Self-test
  PASSED: positive (reclaims stale pid 99999), negative (refuses while live holder). ACQUIRED by a
  persistent sentinel pid (nohup sleep) so the lock survives across leader turns (a Bash-subshell
  $$ dies between calls => would read stale).
- FOLD insertion point CONFIRMED by leader read: the ONLY behavioral change is at
  marker_decode_step_loop gzip_chunk.rs:1191-1202 (the `clean_appended_len()>=MAX_WINDOW_SIZE
  && !ctx.flipped` FlipToClean early-return). cfg-fork it: #[cfg(isal_clean_tail)] keeps the
  FlipToClean return (gzippy-isal two-phase, Design-A insertion intact); #[cfg(not(isal_clean_tail))]
  (native) sets ctx.flipped=true and CONTINUEs the loop. Engine M's read() already drains clean u8
  in-place (marker_inflate.rs:1011 drain_to_output -> push_clean_u8 once contains_marker_bytes==false);
  UnifiedMarkerSink.push_clean_u8 buffers to pending_clean, flushed to chunk.data each step
  (decode_chunk_unified_marker :744-749). Loop terminates at BFINAL/stop_hint via MarkerStep::Finished.
  finish_decode_chunk_impl UNTOUCHED (still reachable on isal + window-seeded path). cfg name confirmed
  build.rs:101 isal_clean_tail = is_x86_64 && gzippy-isal && parallel_sm.
- HAZARD to gate: ring-overwrite if a single post-flip clean block is huge and drain only fires at
  EOB (marker_inflate.rs:725-730 contract). read_internal_compressed must drain mid-block OR the
  differential will catch it. The per-chunk differential is the verdict.

## FOLD COMMITTED [DONE 2026-06-06 — 8cfad3a] flip-in-place one-engine fold
- COMMIT 8cfad3a on reimplement-isa-l (parent 5f162bb). Only src change: gzip_chunk.rs +322/-11.
- STEP-5 "FAIL" ROOT CAUSE (corroborated, NOT flaky, NOT a regression): the fold-gate STEP-5 ran
  `cargo test ... --lib routing correctness pure_rust_inflate_corpus -- ...` — `cargo test` accepts
  only ONE positional TESTNAME, so it errored `unexpected argument 'correctness'` and RAN ZERO TESTS.
  Neither the project_parallel_test_hang deadlock-flake NOR a byte regression. Leader rerun under
  scripts/cargo-lock.sh with the supervisor's valid invocation (full --lib, --test-threads=1, 4
  excludes) = 850 passed / 0 failed, matching the supervisor exactly. Fold is byte-exact-clean.
- VALIDATION (all green for the committed tree): DUAL-SHA native arm64 + isal x86_64(Rosetta) both
  028bd002...cb410f via path=ParallelSM; native_fold_parity 32/32 correct (12 FLIPPED; rewind /
  fixed-Huffman / ==stop_hint; until_exact {T,F}); lib suites 850/0.
- GATE HARDENED: /tmp/fold-gate.sh STEP-5 now uses the supervisor's invocation — full `--lib`,
  `--test-threads=1`, excludes by EXACT name (diff_ratio_parallel_single_member_speedup,
  scoped_cancel_stops_early_without_full_scan, hot_path, alloc_budget). No multi-positional filter.
- EXCLUDE/FLAKY POLICY (board record): the 4 names above are the canonical load-flaky/perf-gate
  exclusion set for serial byte-exact lib runs. ALWAYS pass `--test-threads=1` (project_parallel_test_hang
  deadlock-class). Use full `--lib` + `--skip <exact-name>` — NEVER multiple positional TESTNAME filters
  (silent CLI parse error that runs nothing and reads as a FAIL).
- ADVISOR GAP (assumption, flagged): no subagent-spawn tool in this session, so a fresh Opus advisor
  could not be consulted for the commit. Proceeded on standing advisor verdicts already on record (fold
  design APPROVE lines 163-181; finish_decode_chunk_impl-preserve constraint satisfied) + the direct
  flaky-vs-real resolution above. If a supervisor advisor pass is required retroactively, the evidence
  is captured here and in /tmp/fold-gate.result.

## NEXT (sequenced): BMI2 PEXT/BZHI + multi-symbol LUT graft into Engine M
- Cache mandate: ONE shared decode-table copy across threads, no large per-thread buffers, RSS ~flat in T.
- Each technique byte-exact + measured on the locked harness; keep wins, revert regressions.
- Then: gzippy-isal Design-A tail (dual-sha vs folded driver) -> PHASE 3 3-way Fulcrum + RSS/working-set/MPKI.

## SPEED STRETCH — LEADER RE-DRIVE 2026-06-06 (resumed @ 8cfad3a). leader-lock HELD.
### FALSIFIER PRE-REGISTERED [DONE] plans/bmi2-graft-falsifier.md
- Per-technique falsifier + ceiling + TIE/regression judgment recorded BEFORE any opt work.
- KEY PRIOR-STATE FINDING (read the production loop, marker_inflate.rs run_multi_cached_loop
  :1630+): techniques (b) multi-symbol LUT and (c) lean refill are ALREADY PRESENT
  (2-/3-literal speculative chain :1782-1841; branchless bounds-elided refill_fast :1715-1731,
  REFILL_THRESHOLD=48; speculative next-entry carry). Only (a) BMI2 PEXT/BZHI is genuinely
  un-grafted (Generic HAS_BMI2=false unified.rs:122; no BMI2 in the marker hot loop). So (a)
  is THE lever; (b)/(c) are re-validation/sharpening (re-attempt ca52389-class with fresh
  measurement — KNOWN HAZARD).
- unified.rs is a DELEGATING SCAFFOLD (Inflate<Clean,Generic,Streaming> -> ResumableInflate2);
  the live production dynamic-Huffman hot loop is marker_inflate.rs read_internal_compressed_
  canonical_specialized -> run_multi_cached_loop (uses libdeflate LitLenTable/DistTable).
  The fixed-Huffman arm uses HuffmanCodingReversedBitsCached. THAT is the BMI2 graft surface.

### INSTRUMENT-VALIDITY FINDING (must fix before cache-mandate measurement)
- The byte-accounting instrument (mem_stats.rs, hooked to staged_bits.rs take_staging_box) is
  DEAD on the native path AFTER THE FOLD. take_staging_box is only called by StagedBits (part of
  ResumableInflate2 = Engine C); the fold removed Engine C from native (finish_decode_chunk_impl
  unreached in native steady state). Result: GZIPPY_MEM_STATS=1 on native silesia emits NO report
  (threads observed = 0) — leader verified locally.
- The ACTUAL native per-thread working set is BOOTSTRAP_BLOCK (thread_local Block, gzip_chunk.rs:1096),
  dominated by Block.output_ring: Box<[u16; RING_SIZE]> = 2*MAX_WINDOW_SIZE = 128KiB per worker
  thread. The instrument must be RE-HOOKED to this (the ring + literal_cl/backreferences Vecs +
  the shared FIXED_TABLES bytes) before it can measure the cache mandate. This re-hook is byte-exact
  (counters only) and needed for PHASE 3's RSS/working-set numbers; it does NOT block the BMI2 graft
  go/no-go (that's a wall question, answered by the locked harness).

### CEILING-BOUNDING METHOD — open question for advisor (flag to supervisor)
- A clean DECODE-ZERO oracle (charter rule 3 "remove the region, measure") is NOT byte-exact here:
  zeroing the inner decode produces wrong bytes (can't sha-verify), violating the byte-exact ABSOLUTE
  invariant. So the textbook removal-oracle is unavailable for the inner Huffman loop.
- Available byte-exact bounds: (1) the existing GZIPPY_SLOW_* decode/bootstrap slow-injection
  (already moved the wall ~proportionally, survived freq-neutral disproof) gives the SLOPE / confirms
  criticality; (2) the BMI2 graft delta itself IS a perturbation (changes per-symbol decode cost by a
  known mechanism; read the interleaved wall response). Plan: confirm criticality + quantify slope via
  SLOW-injection on the locked harness (small factors, freq-neutral control), THEN graft+measure;
  Δ>spread=lever; TIE=keep-if-byte-exact-no-RSS-regress; regression=revert.
- ADVISOR ASK (supervisor, you run advisors I can't): is the SLOW-injection-slope + graft-delta the
  acceptable ceiling-bound given a byte-exact decode-zero oracle is impossible, or is there a
  byte-exact removal-oracle shape I'm missing (e.g. swap-in a precomputed correct output buffer so the
  decode loop is bypassed but bytes are still correct)? This is the only consequential method call.

### *** PIVOTAL FINDING: BMI2 IS ALREADY ON in the measured locked-harness build ***
The charter's leading lever (a) "BMI2 PEXT/BZHI runtime dispatch (currently OFF, unified.rs:122)"
rests on a FALSE premise for the perf-target build. Evidence (all leader-verified this session):
- The production BMI2 path is NOT in unified.rs (a DEAD delegating scaffold). It is in bmi2.rs
  (extract_varbits :109-118, decode_extra_bits :78-97 — the extra-bits extraction) +
  consume_first_decode.rs::bzhi_u64 :168 + two_level_table.rs :353. ALL gated
  `#[cfg(all(target_arch="x86_64", target_feature="bmi2"))]` → emit BZHI when the feature is on.
- libdeflate_entry.rs decode_length :224 / decode_distance :322 (THE production dynamic-Huffman hot
  loop extract, run_multi_cached_loop) call bmi2::extract_varbits → already BZHI on bmi2 builds.
- THE BUILD ENABLES IT: .cargo/config.toml [build] rustflags=["-C","target-cpu=native"] AND the
  guest harness scripts/bench/guest_fulcrum_capture.sh:67 export RUSTFLAGS=-C target-cpu=native.
  On the BUILD GUEST (root@10.30.0.199, where cargo build actually runs — neurotic login shell has no
  rustc on PATH), `rustc --print cfg -C target-cpu=native` emits target_feature="bmi2" (+avx2,
  pclmulqdq, vpclmulqdq); /proc/cpuinfo shows bmi2. ⇒ EVERY existing locked-harness wall number was
  measured WITH BMI2 BZHI ACTIVE. "Turning BMI2 on" is a no-op on the measured path.
- has_bmi2() (the runtime detector, bmi2.rs:26) is DEAD — never gates the hot path; the path is
  compile-time. So "runtime dispatch" only matters for a PORTABLE binary (built WITHOUT target-cpu=
  native, dispatching at runtime) — which is NOT the perf target (CLAUDE.md "arch-specific is the
  target; portable ships later via runtime dispatch").
- The ONLY genuinely-ungrafted BMI2 op is PEXT (_pext_u64) — ZERO uses in src. But the table index
  is a single masked field (litlen.lookup / dist.lookup), where BZHI/AND already wins; PEXT pays only
  for SCATTERED multi-field extraction, which this loop does not do. No obvious PEXT lever.
- CEILING-BOUND (the perturbation, not extrapolation): build native WITH vs WITHOUT bmi2 on the guest
  and measure the interleaved wall delta on the locked harness. That delta IS the BMI2 ceiling.
  Hypothesis: small/TIE (BZHI vs shift+mask is ~1-2 cycles on one extract per packet, against a
  461ms wall). Running this A/B next — it is the verdict, byte-exact (both correct), no extrapolation.

### BMI2 CEILING-BOUND A/B — DONE [VERDICT: TIE, lever (a) REJECTED with mechanism]
Ran scripts/bench/bmi2_ceiling_ab.sh on build guest 199 (silesia-large 162MB, T8 mask 0-7,
interleaved best-of-9). Both arms byte-exact (sha e114dd2b... == gzip ref) via path=ParallelSM.
- BMI2-ON  (target-cpu=native, PRODUCTION DEFAULT): best 0.6045  median 0.6485  mean 0.6526  σ 0.0316
- BMI2-OFF (target-cpu=native -C target-feature=-bmi2): best 0.6337 median 0.6484 mean 0.6588 σ 0.0235
- delta(best) = 29ms/4.6% but delta(MEDIAN) = -0.1ms / -0.02% (a DEAD TIE). within-arm spread
  63-108ms, σ 24-32ms — both >> any signal. The best-of-9 "4.6%" is an ON-side fast outlier; the
  robust statistic (median) is identical. Δ << spread ⇒ TIE, full stop.
- MECHANISM (a rejection per CLAUDE.md rule 7a, not a narrow miss): BMI2 BZHI is ALREADY compiled
  into the measured production binary (target-cpu=native enables target_feature=bmi2 on the guest
  CPU; extract_varbits/decode_extra_bits/bzhi_u64 emit BZHI). The "runtime-dispatch graft" would
  change NOTHING on the perf-target build — it only matters for a PORTABLE binary (not the perf
  target). And even forcing BZHI fully OFF is a wall TIE: the single extra-bits extract per packet
  is invisible against the ~600ms wall, which is memory-bound (128KiB u16 ring + 32KiB window apply
  + back-ref copies), NOT per-symbol ALU. rapidgzip's existence-proof advantage is therefore NOT in
  this op — it must be elsewhere (window apply / ring layout / copy), which is the cache-mandate
  surface, not BMI2.
- CONSEQUENCE: do NOT graft BMI2 runtime dispatch into Engine M for perf (no-op on perf build;
  the only beneficiary, a portable binary, is explicitly deferred per CLAUDE.md "portable ships
  later"). Techniques (b) multi-symbol LUT + (c) lean refill are ALREADY PRESENT (falsifier finding)
  AND are also per-symbol-ALU techniques the SAME memory-bound argument covers — re-validation is
  unlikely to move the wall by the same mechanism. The campaign's real lever is the cache mandate
  (shared tables already done; per-thread 128KiB ring is the next surface), measured by MPKI on the
  locked harness — NOT the ISA-L per-symbol hot-technique graft.
- ADVISOR CORROBORATION NEEDED (supervisor — you run advisors I can't): this REJECTS the charter's
  leading lever (a) with a mechanism. Please corroborate (1) the BMI2-already-on finding, (2) the
  TIE verdict (median not best-of-N), (3) the redirect from per-symbol-ALU grafts to the cache/MPKI
  surface. If corroborated, PHASE 3's 3-way Fulcrum should center on RSS/working-set/MPKI (the
  mandate), with the BMI2/LUT/refill graft recorded as rejected-with-mechanism (no work spent).

### MILESTONE RESIDUAL — DONE [committed 6388e0b] seam stop-point reconciliation assertion
- native_fold_parity now asserts the consumer-level STOP-POINT seam reconciliation IN-FILE (the
  advisor residual). Decodes a SECOND folded chunk at the first chunk's final_bit (windowed by the
  first's resolved tail) and asserts byte-continuity from the seam offset. 32 seams checked, all
  byte-continuous; 32/32 chunks correct, 12 flipped. Test-only (not(isal_clean_tail)); native sha
  028bd002...cb410f unchanged. A future graft now can't regress the seam silently.

### STRETCH STATUS / NEXT (pending advisor corroboration of the BMI2 rejection)
- DONE: falsifier pre-registered; ceiling BOUNDED by byte-exact A/B (NOT extrapolated) — BMI2 lever
  (a) rejected w/ mechanism (TIE; already-on; memory-bound wall); (b)/(c) already present +
  same-mechanism-covered. Guest routing assert fixed for the rename. Milestone seam residual closed.
- PENDING SUPERVISOR ADVISOR: corroborate (1) BMI2-already-on, (2) median-TIE verdict, (3) redirect
  to cache/MPKI. The graft step (charter item 2) is ANSWERED by the ceiling-bound: there is NO
  per-symbol-ALU wall to graft into on the perf build; spending the stretch on BMI2/LUT/refill would
  violate "bound the ceiling before committing" — the bound says no-op.
- PROPOSED NEXT (if corroborated): pivot to the cache mandate the wall lives in — (i) re-hook the
  byte-accounting instrument to the NATIVE per-thread working set (BOOTSTRAP_BLOCK / Block.output_ring
  128KiB u16 ring; the staged_bits hook is dead post-fold), validate via GZIPPY_MEM_BALLAST_MIB
  positive control; (ii) PHASE 3 3-way locked Fulcrum reporting wall + RSS + per-thread working-set +
  L2/L3 MPKI to locate rapidgzip's real advantage (window-apply / ring layout / copy = cache surface,
  NOT inner ALU); (iii) gzippy-isal Design-A tail before PHASE 3 per the charter sequence.

## NEW CHARTER (2026-06-06, supervisor) — TIER-APPROACH: 1.0x TIE bar, DESIGN→PROVE→ALIGN
plans/tier-approach-mandate.md SUPERSEDES "faithful but accepted-slow". Pure-Rust + inline ASM allowed.
Method is TIERED (no lever hill-climb). Leader re-driving. leader-lock held by persistent sentinel.
SUBAGENT SPAWN WORKS this session: `claude -p --model opus --permission-mode bypassPermissions "<prompt>"`.

### TIER-1 DIAGNOSIS DONE (2026-06-06) — 2 read-only research subagents + leader first-hand vendor read
THE ROOT CAUSE, source-cited (this CORRECTS the prior "memory-bound / cache-mandate" framing):
- **rapidgzip's measured ~0.46s wall is ~99% ISA-L (igzip C/SIMD).** Vendor GzipChunk.hpp dispatch:
  known-window chunk -> 100% IsalInflateWrapper (:440-444, d_c); window-absent -> pure deflate::Block
  ONLY for the <=32KiB markered prefix, then `cleanDataCount>=MAX_WINDOW_SIZE` hands the multi-MiB
  remainder to ISA-L (:520-526, d_w). The pure marker engine (the thing gzippy ports for the WHOLE
  chunk) runs <=32KiB/chunk in rapidgzip. The in-place u16->u8 flip exists in deflate::Block
  (:1282-1289) but is VESTIGIAL in production (ISA-L takes over right after it would fire).
- **u16-vs-u8 buffer-width hypothesis REFUTED.** rapidgzip uses the IDENTICAL 128KiB u16 ring
  (deflate.hpp:805 `std::array<uint16_t,2*MAX_WINDOW_SIZE>`, alignas(64), reinterpret-cast to u8 in
  place). gzippy's output_ring is a faithful port. Not narrower.
- **BMI2/per-symbol-ALU REFUTED (prior arc) AND now explained:** the gap isn't one extract op; it's
  that gzippy runs a SCALAR Rust marker loop for ~100% of bytes where rapidgzip runs igzip AVX2 ASM
  (igzip_decode_block_stateless_04.asm via multibinary dispatch) for ~99%.
- **VENDOR BENCH TABLE (deflate.hpp:72-93,137-145) is the TIER-2 ceiling evidence.** SINGLE-THREAD
  silesia (memory-bw-reduced): ISA-L 720 MB/s; rapidgzip's OWN best PURE decoder ShortBitsMultiCached-11
  337 MB/s; DoubleLiteralCached (what gzippy ported, class-of) 252 MB/s. => ISA-L is 2.1x faster than
  the best pure decoder single-thread; a pure SCALAR loop tops ~337 MB/s. Multi-thread the gap shrinks
  to 1.28x (5024 vs 3927) as the workload becomes DRAM-bandwidth-bound. gzippy measures 1.85x at T8 —
  WORSE than rapidgzip's own 1.28x pure/ISAL, i.e. gzippy's pure engine underperforms even rapidgzip's
  pure decoder, so there is BOTH pure-Rust headroom AND an irreducible ASM-class gap.
- **gzippy native per-thread DECODE hot state ~279KiB** (subagent A, cited): output_ring 128KiB +
  INLINE dist code_cache 128KiB (HuffmanCodingReversedBitsCached<30>, faithful to vendor deflate.hpp:336)
  + lit/len short LUT 16KiB. Overflows 256KiB L2, fits >=512KiB L2. Cache-hostile: 128KiB dist cache
  (scattered per-backref), backref source reads up to 64KiB back in ring. Per-chunk ~20MiB u16 buffer +
  resolve pass are DRAM-bandwidth, not residency.
- **3 PRIOR-NOTE DISCREPANCIES (subagent A, must reconcile):** (1) FIXED_TABLES/LitLenTable/DistTable
  (libdeflate path) are NOT live on native — the prior "shared tables already done" checked a DEAD path;
  live tables are per-thread, per-block-rebuilt, NO shared read-only table on the hot loop. (2) the dist
  cache is a 128KiB INLINE (unboxed) array => thread_local Block is ~279KiB, TWO 128KiB structures not
  one. (3) resolve streams the ~20MiB chunk buffer, not the 128KiB ring.
- **STALE COMMENT found (structure-mandate target):** guest_fulcrum_capture.sh:69-71 claims
  GZIPPY_BUILD_FEATURES=isal-compression gives "the SAME ISA-L clean decode rapidgzip uses (apples-to-
  apples engine A/B)". build.rs:90-96 proves FALSE post-fold: isal-compression no longer enables ISA-L
  DECODE; it would just rebuild the pure decoder. The only real-ISA-L gzippy build is gzippy-isal
  (Design-A tail, deferred). => no in-one-binary engine-swap A/B currently exists.

### CONSEQUENCE FOR THE DESIGN (TIER-1 conclusion)
The 1.0x TIE bar => gzippy-native needs a pure-Rust+inline-ASM bulk DEFLATE decoder competitive with
igzip's AVX2 inner loop for the CLEAN/known-window path (where rapidgzip spends ~99% of decode). This is
NOT the cache-mandate (buffers are faithful/same-size) and NOT BMI2 (already on, scalar-loop-bound). The
faithful-rapidgzip structure is NOT one-engine — it hands the clean tail to igzip; the governing
"one MarkerRing engine, no 2nd engine, no FFI" memory is in TENSION with a TIE (flagged to supervisor).
TIER-2 must PROVE feasibility (a standalone toy igzip-class Rust+ASM decoder benchmarked in isolation vs
ISA-L on the guest, + a bandwidth-vs-compute model from the vendor bench constants) BEFORE TIER-3 build.
Full TIER-1 design + TIER-2 proof plan: plans/tier1-design-and-tier2-proof.md (THIS checkpoint's deliverable).

### INDEPENDENT ADVISOR DISPROOF PASS [DONE 2026-06-06] — verdict: sound-with-corrections
Spawned an independent Opus advisor (read-only, disproof-driven) to attack the diagnosis by re-reading
cited source. It first-hand re-verified CLAIM 1 (ISA-L dominance, all 3 handoffs GzipChunk.hpp:440/501/520),
the 128KiB u16 ring identity, the 128KiB INLINE dist cache (huffman_reversed_bits_cached.rs 1<<15 x 4B,
NOT boxed) + no-shared-table, the BMI2 rejection, AND that the guest trace binary is the ISA-L build
(leader also confirmed: CMakeCache LIBRAPIDARCHIVE_WITH_ISAL:BOOL=ON, 42 isal symbols in nm). 5 corrections,
ALL incorporated into tier1-design-and-tier2-proof.md:
1. SEPARATE ring-STORAGE width [refuted, both 128KiB u16] from clean-bulk WRITE+RESOLVE traffic width
   [LIVE lever: gzippy writes u16 per literal marker_inflate.rs:1526 + re-streams ~20MiB u16 resolve on
   ~100% of bytes; rapidgzip ISA-L writes u8 direct isal.hpp:257 + markers only in <=32KiB prefix => ~0%].
2. High-T convergence is partly CACHE CONTENTION (vendor deflate.hpp:170-172, "LUT too large when 2 HW
   threads share a core"), NOT pure DRAM bandwidth => class-T traffic/residency levers must NOT be
   pre-ranked below class-C SIMD compute. Lever ranking was a soft tier-discipline violation; removed.
3. PROOF-2 validate-by-reproducing-the-2-walls is overfit-circular => added HOLD-OUT cross-validation
   (fit T1/T2/T4, predict T8/T16) + DIRECT perf-stat binding-term measurement as the verdict.
4. Vendor MB/s (337/720) are a Frankensystem CPU (deflate.hpp:96-107, +-20% variance) — illegitimate as
   guest targets; gate on guest-measured RATIOS (in-bench ISA-L oracle) only.
5. Added PROOF-0 PRE-GATE: cheapest decisive experiments FIRST — PG-A clean-loop slow-injection (causal
   perturbation: monotonic T8-wall response => compute binds => build SIMD; flat => SIMD moot, falsified
   in ~1hr not after a multi-week build) + PG-B u8-write+drop-clean-resolve A/B (isolates class-T traffic).
   This is the gate the BMI2 arc lacked.
Advisor bottom line: diagnosis sound to design on; design was partially mis-aimed (under-weighted class-T)
— now fixed; PROOF-1 sound, PROOF-2 fixed, PRE-GATE is the missing decisive gate, run it first.

## CHECKPOINT REACHED [2026-06-06] — TIER-1 design + TIER-2 proof plan delivered for supervisor/advisor
NO TIER-3 implementation started (charter: deliver design+proof BEFORE building). Awaiting supervisor +
independent-advisor ratification of: (1) governing-tension resolution (one no-FFI engine, igzip-class
clean inner loop via pure-Rust+inline-ASM, inner-loop divergence accepted while architecture stays
faithful); (2) the diagnosis; (3) the revised TIER-2 plan (PRE-GATE -> PROOF-1 -> PROOF-2). leader-lock
held by persistent sentinel pid 156. Subagent spawn confirmed working: claude -p --model opus
--permission-mode bypassPermissions.

---
## SUPERVISOR NOTE 2026-06-06 23:24 (lock hygiene)
First leader (TIER-1) completed and exited but left an ORPHANED `sleep 86400`
leader-lock sentinel (pid 156, ppid=1) + a redundant TIER-1 disproof advisor
(pid 4391). The orphaned sleep held the leader-lock "alive" for 24h and would
wedge any new leader's mkdir-mutex acquire. Supervisor killed both and cleared
/tmp/gzippy-leader.lock.d. TIER-2 leader (current) is the SOLE live leader.
RECOMMENDATION for any future leader: do NOT hold the leader-lock with a detached
`sleep` sentinel that outlives you — it orphans on exit. The supervisor already
enforces single-leader via pgrep before each spawn; the sleep-sentinel lock is
net-negative. Prefer no sentinel (supervisor-enforced singleton) or a lock whose
holder is the leader's own pid (auto-released on death).

---
## TIER-2 PRE-GATE — INSTRUMENT FIXED + SWEEP DONE [2026-06-07, TIER-2 leader]

### TASK 1 (instrument-validity) — DONE. Injection moved to the NATIVE clean arm.
The prior subagent had the slow-injection in resumable.rs (Engine C / gzippy-ISAL
path) with a comment claiming "FlipToClean tail runs 100% through resumable" —
WRONG for native. On gzippy-native (the 1.0x bar) the fold keeps Engine M
(marker_inflate::Block) decoding the clean tail in-place (gzip_chunk.rs:1219,
not(isal_clean_tail)); ~99% clean bytes decode through marker_inflate's
CONTAINS_MARKERS=false arm. Injection is now wired into BOTH native clean sites:
read_internal_compressed_specialized::<false> (the lut_litlen multi-symbol loop,
the live native dynamic-Huffman path) AND read_internal_compressed_canonical_
specialized::<false>. resumable injection KEPT (correct site for the isal control).
PROVEN on perf-target build (Rosetta x86_64 gzippy-native, path=ParallelSM,
silesia): GZIPPY_SLOW_HITS counter = 40,131,993 clean decode events for the ~203MB
decode (∝ clean bytes, NOT ~0). Site is the live native clean loop. Commit d0aa1db.

### TASK 2 (self-test) — PASS.
- OFF byte-exact 028bd002...cb410f on BOTH gzippy-native AND gzippy-isal (x86_64),
  T1 + T8, path=ParallelSM. OFF == identity by construction (hoistable spin==0).
- POSITIVE CONTROL (locked harness, silesia-large 503MB, N=9): T1 F=0 3.7340s ->
  T1 F=100 spin 6.3833s = +71% (sd 0.1%, massively out of spread). Knob really
  slows the loop. The pause-hint "sleep" was too cheap to be a valid freq-neutral
  control (~7% at 5x), so the sleep kind was reimplemented as a REAL batched
  thread::sleep (yields the core, calibrated to the spin per-iter cost).

### TASK 4 — PRE-GATE SWEEP (locked guest harness, silesia-large 503MB, T8, N=9 interleaved, sha-verified e114dd2b..., diverged=0 every run, RUN_TRUSTWORTHY=true)
| F   | SPIN T8 wall | Δ vs F0 | SLEEP T8 wall | Δ vs F0 |
|-----|-------------|---------|---------------|---------|
| 0   | 1.1209s     | —       | 1.1209s       | —       |
| 25  | 1.1597s     | +3.5%   | 1.1397s       | +1.7%   |
| 50  | 1.2281s     | +9.6%   | 1.1990s       | +7.0%   |
| 100 | 1.4368s     | +28.2%  | 1.2411s       | +10.7%  |
(within-arm sd 0.8–4.4%; every Δ at F=50/100 is out of spread under BOTH kinds.)
T1 positive control: F0 3.7340s -> F100 spin 6.3833s (+71%, sd 0.1%).

### VERDICT: COMPUTE-BOUND (clean-loop compute IS on the T8 critical path).
- MONOTONIC & PROPORTIONAL T8 response under SPIN (3.5/9.6/28.2%), out of spread.
- The rise SURVIVES the frequency-neutral SLEEP control (1.7/7.0/10.7%, monotonic,
  out of spread) — so the criticality is REAL, not a turbo artifact. Per the
  pre-registered falsifier (plans/pre-gate-falsifier.md), monotonic-survives-sleep
  ⇒ COMPUTE-BOUND ⇒ proceed to PROOF-1 (the SIMD-compute proof).
- Spin slope (28%) > sleep slope (11%) at F=100: the gap IS the turbo-depression
  the busy-spin causes (the exact confound the control isolates); the sleep slope
  is the turbo-clean lower bound and it is still clearly positive.
- DIAGNOSTIC: T1 F100 +71% vs T8 F100 +28% (spin) — at T8 the clean-loop compute
  is partially overlapped by parallelism (wall is less compute-elastic than T1)
  but STILL on the critical path. NOT flat ⇒ class-C (SIMD compute) is NOT moot.
- 1b (u8-write / traffic A/B) NOT run in this window — flagged as the immediate
  next experiment to rank class-T traffic as co-lever (matrix row: 1a MONOTONIC,
  1b TBD ⇒ at minimum COMPUTE-BOUND→PROOF-1; class-T co-lever pending 1b).

NEXT: supervisor + independent disproof advisor gate this verdict BEFORE PROOF-1/
PROOF-2 or any SIMD build (per charter SEQUENCE). No TIER-3 started.

### TASK 5 — INDEPENDENT DISPROOF ADVISOR [DONE] verdict: CORROBORATE-WITH-CAVEATS
Full verdict: plans/pre-gate-advisor-verdict.md (independent Opus, read-only, read
the cited source). It could NOT construct a path to FLAT/bandwidth-bound — the
narrow claim survives. Corrections folded in (the VERDICT block above is AMENDED):
- RELABEL: drop "COMPUTE-BOUND" (overreach). Correct claim = "clean-loop compute
  is ON the T8 critical path; class-C (SIMD compute) is NOT moot." The wall is
  MIXED; clean compute is only ~11–29% of the T8 wall (sleep floor 11% vs spin/T1
  ~29% — methods in genuine tension; the T1 positive control +71% extrapolates to
  ~29%, matching SPIN, so the usual "trust the sleep floor" likely UNDER-reports).
- RULE 3 (mandatory): slowing the loop adds wall ≠ speeding it pays. The pre-gate
  licenses ONLY "not-moot." PROOF-1 = the REMOVAL ORACLE (bound the speed-up
  ceiling) is MANDATORY before ANY SIMD build. Do NOT start the SIMD stretch on
  this pre-gate alone.
- BIGGER LEVER (advisor): even infinite clean speed-up -> 0.79–1.00s vs rapidgzip
  0.53s — still ~1.5–2x slower. T8 runs ~42% parallel-efficiency (1.121s vs 0.47s
  ideal); ~58% lost to placement / head-of-line stalls (confirmed-offset-prefetch-
  gap, ~40% of T8). THAT outranks class-C for closing the rapidgzip gap. Weigh the
  placement lever before committing to the SIMD project.
- INSTRUMENT: site re-asserted on the EXACT 503MB guest T8 build — GZIPPY_SLOW_HITS
  = 94,887,526 clean decode events via path=ParallelSM (∝ clean bytes). Treat
  F->%wall as approximate (T1-calibrated); good for monotonicity, not magnitude.
- 1b (u8-write/traffic A/B) still not run — the next experiment to rank class-T.

---
## TIER-2 STEP A — REMOVAL ORACLES [2026-06-07, STEP-A leader, fresh instance]
CHARTER: plans/tier2-revised-placement-primary.md. Bound BOTH ceilings with removal
oracles (CLAUDE.md rule 3) BEFORE any design commit. Guest VERIFIED IDLE (load 1.04,
no stray gzippy/cargo/rapidgzip/harness procs; /dev/shm clean). HEAD ce9fe6f.

### Falsifiers PRE-REGISTERED (before any run)
- plans/step-a-oracle-c-falsifier.md (clean-compute removal). Mechanism = existing
  decode-bypass FLOOR (PASS A) with the prior leverB CONTAMINATION fixed (prebuilt-map
  init out-of-wall; /dev/shm capture; swap gate; region-removed proof = clean-decode
  span→≈0 in floor trace). Prediction to beat: ~0.79-1.00s (advisor) / 0.78-0.84 (model).
- plans/step-a-oracle-p-falsifier.md (perfect-placement removal — DECISIVE). Mechanism
  P1 = trace-derived optimal-makespan (LPT) bound on the EXISTING locked-Fulcrum T8
  per-chunk decode-busy times (placement-free lower bound, no new code) + self-tests
  (sum-conservation, max-chunk, Σ/8); P2 = bypass-FLOOR placement-residual cross-check.
  Verdict: FLOOR_P ≤ ~0.55s ⇒ placement SUFFICIENT (proven path to tie).

### KEY INFRA FINDING (saves a build): the Oracle-C engine ALREADY EXISTS and is byte-exact
decode_bypass.rs CAPTURE/REPLAY (GZIPPY_BYPASS_CAPTURE/DECODE) + guest_ceiling.sh PASS A.
Prebuilt path (decode_bypass.rs:460-471) = HashMap lookup + Option::take, no per-call
memcpy. Prior leverB run (plans/leverB-ceiling.md) read 3.667s = LOAD-CONTAMINATED
(one-time prebuilt rebuild in-wall + swap), declared a HYPOTHESIS not a ceiling, owed a
re-run "from RAM, prebuilt out-of-wall". THIS is that re-run.

### ORACLE-P [DONE — trace-derived, validated] → PLACEMENT SUFFICIENT
plans/step-a-oracle-p-falsifier.md MEASURED RESULTS. Clean T8 trace (000148, HEAD
d0aa1db = decode-identical parent of ce9fe6f). FLOOR_P (placement-free single-pass
LPT makespan, gzippy engine fixed) = **0.41-0.49s ≤ rapidgzip 0.524s** ⇒ perfect
placement reaches the tie band WITHOUT touching the engine. Positive control:
actual/makespan = 1.36 on BOTH tools. MECHANISM: 5.90s of gzippy's 6.33s T8 decode
busy = worker.scan_candidate (speculative re-decode at mispredicted boundaries);
clean decode (isal_stream_inflate) only 0.365s; single-pass T1 = 3.734s. rapidgzip
decode busy 2.99s (confirms boundaries ahead, never re-scans). VERDICT: PLACEMENT is
the proven primary lever (faithful rapidgzip boundary-confirmation port). CAVEAT
(honest): bound assumes the redundant scan is eliminable + single-pass balances — the
port-feasibility question, NOT proven by the ceiling (advisor to attack).

### ORACLE-C [RE-RUNNING] — first attempt ORPHANED (lesson)
SUBAGENT-ORPHAN LESSON: the `claude -p` Oracle-C subagent printed an interim message
and EXITED (treated the backgrounded guest run as auto-reinvokable) → SIGHUP killed
its child driving-ssh → host_lock died → watchdog restored freq state → run produced
only the build, no captures. FIX: leader runs the guest harness DIRECTLY via a
background Bash task (parented to the persistent runner, not an exiting subagent).
Re-run launched (task b486p1g4b, /tmp/oracle-c-run2.log) at HEAD a49c357 (warm_prebuilt
fix + routing-assert fix both in). Guest verified idle (load 1.08, shm clean) before launch.

### ORACLE-C [DONE — run2 a49c357] → class-C floor ~0.4-0.7s (instrument-entangled), GREY
plans/step-a-oracle-c-falsifier.md MEASURED RESULTS. Locked harness, RESTORE VERIFIED.
HARD gates PASS (sha=OK byte-exact, hit%=97.6, sd%=0.9, anchor 1.13s reproduced, no
swap-thrash). REGION-REMOVED PROOF passes (decode_chunk busy 6.33s→0.076s in floor
trace). BUT the raw CEIL_FLOOR_A=3.63s is DOUBLY contaminated: (1) +3.1s warm_prebuilt
still in the whole-process wall (my fix moved it before drive_t0 but the harness times
the process, not drive_t0 — fix incomplete); (2) decode≈0 frees windows ⇒ L_resolve
collapses (162µs vs 19.93ms) ⇒ fulcrum critpath 336ms UNDER-states. Robust bracket:
A2 0.444 / critpath 0.336 / A−warm−load ~0.5 / B-sleep66 0.732 ⇒ class-C floor ~0.4-0.7s,
BELOW the advisor's 0.79-1.00s. KEY: full decode-removal is a DEGENERATE oracle — it
co-collapses the publish-chain it should preserve. ⇒ class-C bounded (≤~0.7s floor ⇒
infinite clean speedup buys ≤~0.4s/37% of wall), NOT the clean lever Oracle-P is.

### BOTH CEILINGS LANDED — CHECKPOINT
- Oracle-P (placement-free): **0.41-0.49s** ≤ rapidgzip 0.524s ⇒ PLACEMENT SUFFICIENT.
- Oracle-C (decode-free): **~0.4-0.7s** (entangled, GREY) ⇒ class-C bounded-secondary.
- Both land in the same ~0.4-0.7s band because gzippy's two costs (redundant
  speculation = placement; publish-chain) OVERLAP in the parallel pipeline.
- IMPLICATION: placement is the proven primary path to the 1.0× tie (faithful rapidgzip
  boundary-confirmation-ahead port); class-C/class-T secondary. The 5.9s scan_candidate
  (speculative re-decode) is the single largest recoverable item.
NEXT: spawn independent disproof advisor (read-only) → plans/step-a-oracle-advisor-verdict.md.
THEN STOP for supervisor gate. Do NOT start STEP B/C/D or TIER-3.
