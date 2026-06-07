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
