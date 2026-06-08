# Window-absent decodeBlock 1.6× gap — ATTRIBUTION (owner turn, 2026-06-07, HEAD 7aae6c4a)

## TL;DR (causal, not subtraction)
The window-absent T8 decodeBlock 1.6× gap is **NOT the u16 marker inner loop** and
**NOT u16-width-over-clean-bulk** and **NOT table-build**. It is the **CLEAN u8 tail
decoder** (gzippy's pure-Rust `unified::Inflate<Clean>` / `resumable.rs` engine behind
`StreamingInflateWrapper`, plus `read_internal_compressed_specialized::<false>`),
which runs ~2.3× slower per byte than rapidgzip's ISA-L clean engine. gzippy's u16
MARKER bootstrap is actually FASTER in absolute time than rapidgzip's "custom inflate."

## Apples-to-apples --verbose, T8 unseeded, locked guest REDACTED_IP, taskset 0,2,4,6,8,10,12,14, gov=perf
silesia /root/silesia.gz (211968000 B out), both ~34.5% replaced markers (gzippy 73.0M ≈ rg 73.1M).

| metric | gzippy HEAD (/tmp/gzbuild-head) | rapidgzip 0.16.0 | ratio |
|---|---|---|---|
| decodeBlock SUM | 0.805–0.831 s | 0.4995 s | **1.61–1.66×** |
| marker/bootstrap (u16) decode | body 0.323 s, 73.0M @ 226–235 MB/s | "custom inflate" 0.4748 s | gzippy **0.68× = FASTER** |
| clean tail (u8) decode | ≈0.48 s, ≈139M @ ≈290 MB/s | "ISA-L" 0.2065 s | **≈2.3× SLOWER** |
| Real Decode (÷overlap) | 0.118 s | 0.0845 s | 1.40× |
| Theoretical Optimal | 0.1006 s | 0.0624–0.068 s | 1.48–1.61× |
| std::future::get | 0.084 s | 0.0649 s | 1.29× |
| applying last window | (gather/narrow, separate) | 0.0348 s | — |

gzippy routing (verbose): flip_to_clean=12, finished_no_flip=4, window_seeded=2 (17 chunks).
So 12 chunks decode a u16 marker PREFIX, hit the byte-identical flip
(`distance_to_last_marker_byte>=MAX_WINDOW_SIZE && ==decoded_bytes`,
marker_inflate.rs:1116-1119 == vendor deflate.hpp:1282-1284), then hand off the CLEAN
tail to `finish_decode_chunk_with_inexact_offset` → `StreamingInflateWrapper`
(`unified::Inflate<Clean>` = resumable.rs). 4 chunks never flip (full u16). The clean
tail is the bulk (≈139M of 212M output) and is the dominant decodeBlock term.

## Causal perturbation (MEASURED ΔdecodeBlock SUM, not subtraction; freq-neutral control)
Existing in-tree knobs: `GZIPPY_SLOW_MARKER_MODE` injects per-decode-event work in the
u16 marker careful loop (`marker_spin_iters`); `GZIPPY_SLOW_MODE` injects in the CLEAN
u8 path — wired into BOTH `marker_inflate.rs <false>` AND `resumable.rs:1199` (the
`unified::Inflate<Clean>` engine that decodes the post-flip clean tail). Baseline
decodeBlock 0.831 s.

| inject | kind | decodeBlock SUM | Δ vs baseline | body_ms (marker) |
|---|---|---|---|---|
| baseline | — | 0.831 s | — | 323 |
| MARKER +100% | spin | 0.965 s | +134 ms | 483 (151 MB/s) |
| MARKER +100% | sleep (control) | 0.973 s | +142 ms | 505 (145 MB/s) |
| CLEAN +100% | spin | 1.025 s | **+194 ms** | 312 (unchanged) |
| CLEAN +100% | sleep (control) | 1.079 s | **+248 ms** | 311 (unchanged) |

Spin≈sleep on both ⇒ deltas are real, not turbo artifacts (freq-neutral control
survives). CLEAN inject leaves the marker body untouched (312 ms) yet adds the LARGER
ΔdecodeBlock (+194/+248 ms) ⇒ the clean tail is the dominant decodeBlock term and is
on the critical SUM. MARKER inject (+50% self-time, +160 ms body) adds only +134 ms —
the marker path is on the SUM but is the SMALLER term and is already faster than rg's
custom inflate.

## Attribution verdict among the prompt's three candidates
- **(a) marker inner loop — REFUTED as the prime term.** gzippy's u16 marker bootstrap
  is 0.323 s (226–235 MB/s) vs rg custom inflate 0.4748 s — gzippy is FASTER here. The
  `'mfast` multi-cached u16 loop (deflate.hpp:1585-1666 port) + `emit_backref_ring::<true>`
  backward-scan are competitive. Slowing it adds the smaller SUM delta.
- **(b) u16-width over the clean bulk — REFUTED.** The flip threshold is byte-identical
  to vendor; both gzippy and rg decode the SAME u16 prefix then flip. gzippy does NOT
  run u16 over the clean bulk — post-flip it routes the clean tail to a u8 engine
  (StreamingInflateWrapper). The bulk (139M) is decoded u8, not u16.
- **(c) table-build — REFUTED.** Shared with the clean path; the window-seeded clean
  path (seedfull) TIES rg (0.128 s), so table-build is not 1.6×.
- **ACTUAL CAUSE: the CLEAN u8 tail decoder is ~2.3× slower than ISA-L.** gzippy's
  pure-Rust `unified::Inflate<Clean>` (resumable.rs) clean engine at ≈290 MB/s/thread
  vs rg's ISA-L clean ≈ 0.207 s. This is the governing-memory's "u8-direct clean path"
  — but the gap is the u8 decoder's RAW SPEED vs ISA-L, not a u16-over-clean-bulk
  shortcut. rg hits ISA-L (hand-tuned C) for its clean tail; gzippy-native uses a
  pure-Rust LUT engine.

## Structural ROOT CAUSE (source-verified — the measured build's clean tail is pure-Rust, NOT ISA-L)
Two decode topologies (build.rs:101 `isal_clean_tail = is_x86_64 && has_gzippy_isal && parallel_sm`):

- **gzippy-ISAL build (`isal_clean_tail`) = the measured /tmp/gzbuild-head:** TWO-PHASE.
  Engine M (`marker_inflate::Block`, u16) bootstraps until 32 KiB clean → `FlipToClean`
  (gzip_chunk.rs:1397-1410) → **Engine C = `StreamingInflateWrapper` =
  `unified::Inflate<Clean>` = pure-Rust `resumable.rs`** decodes the clean tail. The
  `SLOW_MODE` +194/+248ms landed HERE (resumable.rs:1199), confirming this is the
  dominant decodeBlock term. **Despite the "isal" name, the clean tail is pure-Rust
  resumable, NOT ISA-L FFI.** resumable.rs:1182-1192 says so outright.

- **gzippy-NATIVE build (`not(isal_clean_tail)`, the 1.0× bar):** FOLD — Engine M keeps
  decoding the clean tail in-place via `read_internal_compressed_specialized::<false>`
  (gzip_chunk.rs:1411). Different clean engine again.

**rapidgzip's WITH_ISAL build hands its clean tail to REAL ISA-L (deflate.hpp:1452-1453
`HuffmanCodingISAL`, verbose "decoding with ISA-L" 0.2065s).** gzippy's "isal" build
does NOT — it runs a pure-Rust resumable engine (~290 MB/s) that is ~2.3× slower than
ISA-L at the equivalent point. THAT is the 1.6× decodeBlock gap, end-to-end.

Per [[project_faithful_unified_decoder_over_perf]] (GOVERNING): the faithful target is
ONE u8-direct clean engine at ISA-L speed continuing on the same cursor (no two-phase
hand-off to a second engine). The pure-Rust clean engine being slower than ISA-L is the
known engine-primitive PLATEAU (prior turn: VAR_VI ≈0.6× ISA-L as a standalone
primitive, advisor-upheld). So this gap reaches the wall and the faithful fix is an
ISA-L-speed pure-Rust u8 clean decoder OR (faithful-isal path, charter goal #2) actually
route the gzippy-ISAL clean tail through ISA-L FFI.

## SCOPED FIX (do NOT start — supervisor gate; bound by a removal oracle first)
Two faithful candidates, each pre-bounded by the Phase-0 ISA-L engine oracle
(GZIPPY_ISAL_ENGINE_ORACLE, already ties seeded — re-use as the engine-removed ceiling,
NEVER the slow-down slope):
1. **gzippy-faithful (isal, charter goal #2):** route the post-flip clean tail through
   real ISA-L FFI on the `isal_clean_tail` build (= rg's WITH_ISAL build exactly). This
   is the reference baseline and the cleanest test of "is the clean-engine speed the
   binder." Largely the Phase-0 oracle made permanent on the FlipToClean tail.
2. **gzippy-native (charter goal #1, the 1.0× bar):** the clean fold already uses
   `read_internal_compressed_specialized::<false>` (VAR_V fast loop). Its ceiling is the
   ~0.6× ISA-L primitive plateau — so a pure-Rust 1.0× clean engine needs the
   inner-Huffman work the charter authorizes (BMI2 PEXT, wider multi-literal, ISA-L LUT
   parity). Bound with the oracle before a work-stretch.

CAVEAT for the advisor: the measured clean-tail ≈0.48s is a SUBTRACTION (decodeBlock −
marker body). The CAUSAL evidence is the SLOW_MODE A/B (+194/+248ms, marker body
unchanged) which lands in resumable.rs — that is the load-bearing proof the clean tail
is the dominant decodeBlock term. The 2.3× ratio itself rests on rg's "ISA-L 0.2065s"
vs the subtraction; treat the 2.3× as the hypothesis and "clean tail is the bigger
decodeBlock term, NOT the marker loop" as the causally-proven claim.
