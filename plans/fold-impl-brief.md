# SUBAGENT BRIEF — flip-in-place fold (native) + per-chunk differential gate

You are an implementation subagent for the gzippy rapidgzip port. Repo:
/Users/jackdanger/www/gzippy-reimplement-isal  (branch reimplement-isa-l @ 5f162bb).

## GOAL (one behavioral change + one new permanent gate)
Make the NATIVE decode path (gzippy-native; cfg `not(isal_clean_tail)`) fold the
two-engine clean tail into ONE engine: Engine M (marker_inflate::Block) keeps
iterating in-place past the 32 KiB clean threshold instead of handing off to
Engine C (StreamingInflateWrapper). The gzippy-isal path (cfg `isal_clean_tail`)
KEEPS the current two-phase FlipToClean handoff unchanged.

## THE EXACT CHANGE (surgical — cfg-fork ONE decision point)
File: src/decompress/parallel/gzip_chunk.rs, fn `marker_decode_step_loop`, the
FlipToClean early-return at ~line 1191-1202:

```rust
if output.clean_appended_len() >= MAX_WINDOW_SIZE && !ctx.flipped {
    ctx.flipped = true;
    let end_bit_offset = next_block_offset;
    ctx.current_bit_offset = end_bit_offset;
    return Ok((MarkerStep::FlipToClean { end_bit_offset, window_len: MAX_WINDOW_SIZE }, false));
}
```

cfg-fork it:
- `#[cfg(isal_clean_tail)]` arm: KEEP the code above verbatim (gzippy-isal two-phase;
  this is the Design-A insertion point — do NOT change its behavior).
- `#[cfg(not(isal_clean_tail))]` (native) arm: set `ctx.flipped = true;` and FALL
  THROUGH / do NOT return — let the loop continue decoding this and subsequent blocks
  on the SAME ctx cursor. Engine M's `read()` already drains clean u8 in-place
  (marker_inflate.rs:1011 -> drain_to_output -> push_clean_u8 once
  contains_marker_bytes==false). The loop will terminate naturally at BFINAL
  (MarkerStep::Finished, marker_inflate stride :1293) or stop_hint (:1222).

WHY this is byte-safe (already verified by leader, but re-confirm):
- `UnifiedMarkerSink::push_clean_u8` (gzip_chunk.rs:655) buffers into pending_clean;
  decode_chunk_unified_marker (:744-749) flushes pending_clean to chunk.data +
  bumps clean_data_count each step. So post-flip clean bytes already flow to chunk.data.
- Ring-overwrite is structurally prevented: read_internal_compressed_specialized
  caps n_max_to_decode to RING_SIZE - MAX_RUN_LENGTH (marker_inflate.rs:1258) and
  returns < n_max so read() drains between calls.
- `ctx.flipped` still gates the threshold to fire once (sets true), preventing
  re-entry; the `!ctx.flipped` guard keeps that check from re-evaluating each block.

DO NOT touch finish_decode_chunk_impl (gzip_chunk.rs:354) — it stays reachable on the
isal path and the window-seeded path (:608-627). DO NOT change Engine C.

CAUTION: after the fold, on the native path FlipToClean is never returned, so the
match arm `MarkerStep::FlipToClean` in decode_chunk_unified_marker (:757) becomes
dead on native but LIVE on isal. Keep the arm (cfg or #[allow]); do not delete it —
it must stay live for isal_clean_tail. If the native build warns dead-code, gate the
arm or add a targeted allow; do NOT silence broadly.

## DELIVERABLE 2 — HARDENED per-chunk differential gate (PERMANENT)
The isal-parity-gate-mandate (plans/isal-parity-gate-mandate.md) requires the
STEP-1b pure-vs-isal tail differential (8d026a8) be HARDENED and run against the
FOLDED native driver. Find the existing differential test (search:
`grep -rn "isal_clean_tail" src/ | grep -i "test\|differ"` and look in
src/tests/ or #[cfg(all(test, isal_clean_tail))] modules). Then:
- HARDEN it to exercise: the gzip_chunk.rs:486 rewind path (a non-final/non-fixed
  EOB-header rewind to last_eob_pos), a fixed-Huffman no-rewind case (:481 not_fixed
  false), and the `next_block_offset == stop_hint` boundary (:484). Use real silesia
  chunks; assert IDENTICAL (decoded bytes, committed length, final_bit handoff,
  per-chunk CRC, block boundaries) for both until_exact true AND false.
- ALSO add a NATIVE-side assertion: the FOLDED native decode of a chunk produces
  byte-identical decoded output + final_bit + CRC to the (pre-fold-equivalent)
  two-phase decode of the SAME chunk. This is what makes the gate catch a fold that
  silently diverges. Keep it as a permanent #[cfg]'d test.

If you cannot fully wire the native-vs-isal cross-check in this pass, IMPLEMENT THE
FOLD + the silesia DUAL-SHA gate, and REPORT precisely what differential hardening
remains (file:line of the existing test, what you added, what's left). Do not fake it.

## BUILD / GATE DISCIPLINE (MANDATORY)
- EVERY cargo/rustc/test MUST go through `scripts/cargo-lock.sh <cmd>` (global mkdir
  mutex — concurrent builds are forbidden). Before building, `pgrep -x cargo; pgrep -x
  rustc` to confirm none running.
- DUAL-SHA GATE (both must emit silesia sha256
  028bd002c89c9a909ccdbc2af0a223de285348edb014ccc8e27d297f52cb410f via path=ParallelSM):
  1. NATIVE (arm64): `scripts/cargo-lock.sh cargo build --release --no-default-features
     --features gzippy-native`
  2. ISAL (x86_64 via Rosetta cross-compile — native cargo, NOT `arch -x86_64 cargo`):
     `scripts/cargo-lock.sh env CARGO_BUILD_RUSTFLAGS="-C target-cpu=x86-64" cargo build
     --release --target x86_64-apple-darwin --no-default-features --features gzippy-isal`
- Find the silesia .gz corpus (search ~/ or repo for silesia*.gz; check
  reference_compression_corpus memory: https://jackdanger.com/squishy/). Decompress
  via the built binary, sha256 the output, assert == reference. Assert GZIPPY_DEBUG=1
  prints path=ParallelSM.
- Run the native lib test suites single-invocation, --test-threads=1, timeout-wrapped:
  routing::, correctness, pure_rust_inflate_corpus:: (includes silesia), index (the
  --index/scan second decode path). Skip load-flaky (not_slower/diff_ratio/scoped_cancel/
  hot_path/alloc_budget). Run the hardened differential test under the isal x86_64 build
  (it's cfg(isal_clean_tail)).
- clippy + cargo fmt --check clean (hooks require fmt).

## OUTPUT (report back, DO NOT commit unless gate is fully green)
Report: the exact diff (file:line), DUAL-SHA result (both shas + path=ParallelSM
confirm), test pass counts, differential-hardening status (what you added / what
remains), any divergence mechanism you found. If gate green, you MAY commit on the
branch with a message ending the Co-Authored-By line; otherwise leave uncommitted and
report blockers. Be precise and honest — a TIE/divergence is a finding, not a failure.
