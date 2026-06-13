# Liar-Sweep Advisor Gate — Verdict (read-only Opus gate)

Branch reimplement-isa-l, HEAD d56cb0f5. Reviewing the documentation-correctness sweep's
comment+memory edits, then hunting for what it missed.

ESTABLISHED TRUTH re-verified independently at HEAD:
- Cargo.toml:84 `gzippy-isal = ["pure-rust-inflate", "isal-compression"]` ✓
- build.rs:110 `isal_clean_tail = is_x86_64 && has_gzippy_isal && parallel_sm` ✓
- gzip_chunk.rs:161 `isal_engine_oracle_enabled()` default arm `_ => cfg!(isal_clean_tail)` ✓
  (blamed to **19add96c**, 2026-06-08 13:02 "ship real-ISA-L clean tail as PRODUCTION routing")
- gzip_chunk.rs:669 production call `if allow_isal && isal_engine_oracle_enabled()` ->
  finish_decode_chunk_isal_oracle -> :275 `decompress_deflate_from_bit_into` ✓
So at HEAD, gzippy-isal's production single-member clean tail runs REAL ISA-L FFI by default.
"No C-FFI in the decode graph / pure-Rust sole" is a gzippy-NATIVE-only property. CONFIRMED.

---

## Item 1 — src/decompress/mod.rs + single_member.rs (7 comment edits): **SOUND**
Each edit scopes a former universal "pure-Rust sole / no-FFI" claim to
"ParallelSM pipeline sole; pure-Rust on native, ISA-L clean tail on isal", and is
accurate at HEAD:
- mod.rs:156-179 classify_gzip comment — correct: routes to ParallelSM, native pure-Rust,
  isal clean tail via FFI; one-shot C-FFI only on not(parallel_sm). Matches gzip_chunk.rs:650.
- mod.rs:432-437, :481-485 (one-shot doc), :778-784 + :907-914 (two tests) — all consistent.
  The distinction drawn (never a *one-shot* C-FFI; isal clean tail IS C-FFI) is exactly right.
- single_member.rs:28-33 (MIN_PARALLEL_SIZE history) and :171-174 — correct, and the new
  citation `gzip_chunk.rs finish_decode_chunk_impl` is the right production entry.
No overcorrection: edits did not claim isal is FFI-free; they did not erase the native FFI-free
fact. Honest-history note ("Removed 2026-06-04 ...") preserved.

## Item 2 — memory/project_task8_ffi_removal_state.md: **SOUND**
(a) Multi-member OPEN-GAP -> RESOLVED-2026-06-08 annotation (lines 140-145): VERIFIED.
   - f7970c99 "pure-Rust trailing-member decode on the single-member path" is real.
   - d56cb0f5 merge "closes the no-FFI multi-member gap" is HEAD.
   - test_concatenated_members_large_first_member_no_truncation is a plain `#[test]`
     (correctness.rs:1447-1448), un-ignored. ✓
(b) Supervisor frontmatter "SCOPE CORRECTION" (line 3) + body "READ-FIRST" (lines 12-18):
   ACCURATE. Correctly attributes ISA-L re-add to 19add96c, scopes pure-Rust-sole to
   gzippy-native, cites finish_decode_chunk_impl -> decompress_deflate_from_bit_into.
   The dated 2026-06-04 DONE/diagnosis history below (lines 20-184) is PRESERVED verbatim
   (annotations are added as `>` blockquotes above the original text, not rewrites). ✓

## Item 3 — MEMORY.md task8 index line + project_scope.md: **SOUND**
- MEMORY.md:10 — "multi-member CLOSED 2026-06-08 ... f7970c99/d56cb0f5; ignored test
  re-enabled": accurate.
- project_scope.md:17 — zlib_format.rs absent at HEAD: VERIFIED (find returns nothing).
- project_scope.md:18 — docs/rapidgzip-port-reference.md + CLAUDE.md "Cutover Goal"/
  "CUTOVER POLICY" deleted by b1bb7e12: VERIFIED (the doc is in b1bb7e12's deleted-file
  list; CLAUDE.md has 0 matches for those headings). NOTE: docs/ DIRECTORY still exists at
  HEAD (other files); the note only claims the two specific pointers were deleted — that
  reading is correct, not "the whole docs/ dir was deleted." No fix needed.

---

## HUNT — what the sweep MISSED

### A. Other memory files asserting no-FFI as CURRENT+UNIVERSAL
- **project_implementation_task_pure_rust_port.md — SOUND (honest goal-statement, scoped
  by banner).** "DELETE the C-FFI ... Pure-Rust inflate is the SOLE decode engine" is a
  2026-06-04 USER DIRECTIVE / task framing, not a current code-state assertion, and the
  file carries the standard "4 days old ... verify against current code" banner. A reader
  is told this is the goal, not HEAD reality. Lines 129-148 ("ONE-CONFIG + FFI FULLY
  DELETED 2026-06-04") are dated and self-scoped to that date. Would benefit from a
  one-line "isal build re-added FFI clean tail (19add96c)" pointer for symmetry with
  task8, but it is NOT a current-false lie — do not block on it.
- **project_faithful_unified_decoder_over_perf.md — SOUND.** The GOVERNING note (lines
  10-29) EXPLICITLY models BOTH builds: gzippy-native = u8 no-C-FFI; gzippy-faithful(isal)
  = "+ ISA-L via C-FFI ... faithful WITH_ISAL." This file is the one that gets the
  two-build distinction RIGHT. No misreading risk.
- **feedback_no_innovation.md — SOUND.** Architecture-vs-inner-loop scope; makes no
  no-FFI/pure-Rust-sole universal claim. Untouched correctly.
- MEMORY.md index lines for the above (5, 12, 13) — consistent with their targets. SOUND.

A reader of A-group files would NOT mis-take "isal is also FFI-free": faithful_unified
states the opposite explicitly; implementation_task is banner-scoped goal text.

### B. Renamed-file references in DATED history (isal_lut_bulk.rs, isal_huffman_pure.rs)
- **AGREE: keep as history.** These appear inside dated 2026-06-04 diagnosis blocks
  (project_implementation_task_pure_rust_port.md:95-204, project_task8:96-114). They are
  retrospective findings ("decode_block benchmarks ...", "isal_huffman_pure.rs:1022 is a
  faithful transliteration"), not current-tense recommendations a reader would act on with
  a stale path. The file-rename does not turn these into lies; they are honest snapshots.
  No CURRENT-tense recommendation in the reviewed files keys off a stale renamed filename.
  SOUND to leave.

### C. NEW lie introduced by the edits: NONE in the reviewed source/memory edits.
The edits are conservative scopings; none overcorrected honest history.

---

## *** FIX-NEEDED (HIGH VALUE — the sweep's biggest MISS) ***

### build.rs:105-109 is itself a LYING COMMENT, current-false at HEAD, and UNEDITED.
The comment reads:
> "BOTH topologies decode the clean tail in PURE RUST. Real ISA-L FFI for the clean tail
>  is reachable ONLY under the measurement-only env oracle `GZIPPY_ISAL_ENGINE_ORACLE=1`
>  (gzip_chunk.rs:539) — it is NOT on any production path. (Prior comment claimed this cfg
>  routed through 'REAL ISA-L FFI'; that was stale/aspirational ...)"

Timeline (git blame, definitive):
- **9cde0b4f** (2026-06-07 18:57) wrote this build.rs paragraph.
- **19add96c** (2026-06-08 13:02, *later*, "ship real-ISA-L clean tail as PRODUCTION
  routing") flipped `isal_engine_oracle_enabled()` default to `cfg!(isal_clean_tail)` =>
  on gzippy-isal, ISA-L IS the production clean-tail default with the env var UNSET.

So the build.rs comment is now FALSE in two ways:
1. "BOTH topologies decode the clean tail in PURE RUST" — FALSE: gzippy-isal production
   decodes the clean tail through REAL ISA-L FFI (gzip_chunk.rs:650-684, the production
   comment there says exactly this).
2. "Real ISA-L FFI ... reachable ONLY under GZIPPY_ISAL_ENGINE_ORACLE=1 ... NOT on any
   production path" — FALSE: it's the build DEFAULT on gzippy-isal (env-unset).
3. Bonus stale citation: "gzip_chunk.rs:539" points at a mid-signature line (a parameter
   `configuration: ChunkConfiguration`), not the oracle.

This is precisely the sweep's stated target ("a note that asserts something CURRENTLY TRUE
which is FALSE at HEAD") and it DIRECTLY CONTRADICTS the ESTABLISHED TRUTH the sweep relied
on. It is also the same lie the mod.rs/single_member.rs edits fixed elsewhere — build.rs
was left in the pre-19add96c state. The sweep's source pass covered decompress/ but not
build.rs.

RECOMMENDED FIX (for the editing agent, not me — I am read-only): rewrite build.rs:103-109
to match gzip_chunk.rs:141-164 + :650-657 — e.g.:
  // When OFF (gzippy-native, and arm64 always): the FOLD — Engine M keeps decoding the
  // clean tail in-place in PURE RUST. When ON (gzippy-isal): the clean tail decodes via
  // REAL ISA-L FFI by default (isal_engine_oracle_enabled() defaults cfg!(isal_clean_tail);
  // GZIPPY_ISAL_ENGINE_ORACLE=0 forces pure-Rust for the differential gate). Faithful
  // rapidgzip WITH_ISAL (gzip_chunk.rs finish_decode_chunk_impl / :669).
and drop/repoint the stale ":539" citation.

---

## Summary of per-item verdicts
| Item | Verdict |
|------|---------|
| 1. mod.rs + single_member.rs (7 edits) | SOUND |
| 2. project_task8 (multi-member resolved + scope note + history preserved) | SOUND |
| 3. MEMORY.md task8 index + project_scope.md | SOUND |
| A. implementation_task / faithful_unified / no_innovation | SOUND (goal-scoped, banner'd) |
| B. renamed-file refs in dated history | SOUND (keep as history) |
| C. new lies from the edits | NONE |
| **MISS. build.rs:105-109 "BOTH ... PURE RUST / not on any production path"** | **FIX-NEEDED** |

The sweep's edits are all SOUND and correctly scoped; it did NOT overcorrect honest history.
The one material gap is build.rs:105-109, a current-false comment the sweep didn't reach —
falsified by 19add96c making ISA-L the gzippy-isal production clean-tail default.
