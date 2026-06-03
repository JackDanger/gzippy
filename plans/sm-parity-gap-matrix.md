# Single-member parallel: gzippy vs rapidgzip gap matrix

**Ground truth (locked Fulcrum, `e899062`, T8, silesia-large):** gzippy **~773 ms** trace wall vs rapidgzip **~474 ms** (**~1.63×**). Publish-chain binds (`L_resolve` ≈ **20 ms × ~39** ≈ wall). Both tools have **~31%** static replaced-marker fraction; gzippy has **~90%** runtime window-absent decode starts (structural prefetch depth, not over-marking).

**Verdict rule:** progress = sha-verified gzippy MB/s ↑ and interleaved ratio ↓ by more than sample spread (`scripts/bench/run_locked_fulcrum.sh`). Fulcrum proposes; **causal perturbation** (slow-inject / removal oracle) disposes.

---

## Gap matrix

| ID | Runtime difference | Measured signal | Closure design | Status |
|----|-------------------|-----------------|----------------|--------|
| **A** | Total wall | 1.63× LOSS | Sum of rows below; re-Fulcrum after each | Ongoing |
| **B** | Publish-chain binds | `fulcrum model` residual ≈ 0 | Reduce **C** + **E/F** | Confirmed |
| **C** | `L_resolve` per link | mean **~20 ms**, p95 **~121 ms** | **Design D** worker resolve-ahead | **Measured TIE** (inert: 0 exact marker chunks in cache @ handoff) |
| **D** | Consumer wait (`block_fetcher_get` / `future_recv`) | ~89% wall WAIT | Symptom of decode supply + **C**; not core steal (trisection TIE) | Follow **C**, **E/F** |
| **E** | Clean inflate (`d_c`) | `worker.stream_inflate` **~2×** busy vs rg `isal_stream_inflate` | Match-source `_mm_prefetch` in resumable LENGTH path (libdeflate mirror) | **TIE** trace +3.9% (783 vs 754 ms, within spread); interleaved TIE |
| **F** | Window-absent bootstrap (`d_w`) | `worker.bootstrap` **~2121 ms** busy; +30% wall @ +100% bootstrap slow-inject | `emit_backref_ring`: short copy unroll, contiguous marker scan, cold `classify_and_delay`; causal **confirmed** | Open (Fulcrum trace **TIE** 758 vs 754–759 ms) |
| **G** | Runtime WA% vs static | 90% vs 31% | **Accept**; make WA cheap (**F**, **K**), don't cap prefetch (D6 refuted) | Policy |
| **H** | Partition-seed key mismatch | 37/38 WA = KEY-MISMATCH | **Design H** handoff decode via `get_predecessor(stop_hint)` on speculative prefetch | **Measured TIE** (`HANDOFF_DECODE_CLEAN_OK=0`; handoff key not in map at decode start) |
| **H′** | Pred @ `get_predecessor(start)` | — | Decode at `pred_key < start` + trim to partition seed (vendor spacing fix) | **Reverted** (`5755772`): breaks `test_prefetch_next_filesize_accept_fires` on 56 MiB low-entropy fixture (`Stored block len=0`); needs chain-valid pred only |
| **E′** | Clean tail path split | resumable wrapper on clean finish | **P1** `finish_decode_chunk_bulk_lut` (`GZIPPY_ISAL_PURE_BULK`, default ON) | **Shipped** `5755772`; re-Fulcrum pending |
| **I** | Boundary trial cost | `scan_candidate` **+1373 ms** busy | Tail prefilter + no double-bootstrap (`e899062`); Kraft pre-reject | Partial |
| **J** | `pool.pick` | **+618 ms** busy | Fewer tasks (**I**, **C**); `pick.wait` vs `pick.lock` trace | Open |
| **K** | Marker resolve tax | apply+narrow; rg `apply_window` **~238 ms** busy | Fused path ≥16 KiB; **C** moves resolve off consumer | Partial |
| **L** | Consumer output memcpy | **~100 ms** T8 ceiling (null-consumer oracle) | writev ON; pipe vmsplice N≥21; port rg segment write | Partial |
| **M** | Buffer / store-walks | **~3.26×** dTLB walks; RSS gap | Segment `data_with_markers` @ 128 KiB (O2) | Open |
| **N** | Stalls not instructions | 0.95× insn, 1.42× cycles | Fix **E/F/M**, not “do less work” | Confirmed |
| **O** | T scaling | Gap 1.10→1.54× T1→T16 | `N·L_resolve` serial; needs **C** + faster **d_w** | Expected |
| **P** | Frontier priority | −1 on-demand / −2 post-process | Shipped; insufficient alone | Done |
| **Q** | Eager post-process on consumer stall | +195 ms | **Refuted** — wrong thread | Dead |
| **R** | Bound prefetch depth | Consumer wait explodes | **Refuted** (D6) | Dead |

---

## Closure designs (detail)

### C — Publish-chain / `L_resolve` (Design D)

**Vendor:** `queuePrefetchedChunkPostProcessing` — when predecessor window is confirmed, `applyWindow` runs on the **thread pool**, not the consumer stall.

**gzippy (2026-06-03):** `GZIPPY_RESOLVE_AHEAD=1` — resolve on prefetch when handoff key is published:

- Trigger: `resolve_ahead_prefetch_at_handoff` after confirmed `insert_owned_none` / promote (not at decode return).
- Gate: `max == encoded` (exact chunk only), `max == handoff_key`, `contains(handoff_key)`, markers present.
- Runs `resolve_chunk_markers_on_chunk` + tail publish; sets `chunk.markers_resolved`.
- Consumer skips `wait_replaced_markers`, `window_publish_marker`, and pool post-process for that chunk.

**Ship-gate:** Locked Fulcrum N≥9; `RESOLVE_AHEAD_OK` > 0; wall ↓ > spread; routing + corpus tests green.

**Disproof:** Wall flat, only worker busy ↑ → decode-bound, not resolve-bound.

### E/F — Decode engine

**Difference:** Pure-Rust resumable inflate + Rust marker bootstrap vs rg ISA-L-class paths.

**Plan:** Inner-loop parity (authorized); causal `GZIPPY_SLOW_BOOTSTRAP` after each change. Do **not** reduce WA% by prefetch caps.

### I — Boundary search

**Plan:** Cheap rejects before `decode_chunk_window_absent`; never `get_predecessor` in `try_speculative_decode_candidate` (CRC hazard, proven).

### K — Resolve implementation

**Plan:** Keep ≥16 KiB fused path; share `resolve_chunk_markers_on_chunk` between pool post-process and resolve-ahead.

### L — Consumer output

**Plan:** writev (file); vmsplice pipe with N≥21; study rg `DecodedData` segment write path.

### M — Memory granule

**Plan:** O2 segment `data_with_markers`; ship-gate wall TIE-or-win, not RSS-only.

---

## Execution order

1. **F/E** (bootstrap + clean inflate) — only remaining wall lever on silesia (Fulcrum RATE-bound).
2. Revisit **H** only if early-publish fraction rises (handoff needs `contains(handoff_key)` at worker decode time).
3. **L/M** (output + segments) — bounded ceilings.
4. **I/J** (trials + pool) — incremental.

---

## Process alignment with rapidgzip (what “match” means)

**Already aligned (structure):** prefetcher → worker pool → `WindowMap` → in-order consumer;
per-chunk `decodeChunkWithRapidgzip` shape (marker phase → stream finish on same chunk);
speculative boundaries; post-process `apply_window`; CRC/ISIZE verify.

**Still different (why instructions ≠ rapidgzip):**

| Area | rapidgzip | gzippy today | Close via |
|------|-----------|--------------|-----------|
| Marker bootstrap | Fast C/ISA-L-class path in unified decoder | `deflate_block::Block` pure Rust (~2× busy) | **P2 unified decoder** (one fast loop with marker emit mode) or speed `deflate_block` to parity |
| Clean inflate | ISA-L stream | `ResumableInflate2` (~2× `stream_inflate` busy) | **P1 inner loop** (libdeflate techniques, yield tax) per `unified-decoder.md` |
| Publish / resolve | Pool post-process when window ready | Mostly consumer + `L_resolve` | Design C (TIE so far), consumer output (L) |
| Chain accept | Range + invariant | `max == decode_start` guard | Full chain invariant (refreshed-plan §1.4) if guard-rejects still fire |
| Buffer motion | Segmented write in place | `narrow_markers_in_place` + segmented `data` | Memlife/L/M; verify on trace |

**Not the plan:** two-pass scan; arm64-disable parallel SM (obsolete with `pure-rust-inflate`);
treating Fulcrum Δbusy as verdict without perturbation.

**Build order to “match process”:** (1) measure with counters + Fulcrum, (2) P1 clean decoder
wall win, (3) P2 same decoder for bootstrap (delete hot `deflate_block` path), (4) chain +
resolve scheduling, (5) output path.

---

## References

- [`docs/production-paths.md`](../docs/production-paths.md) — routing truth
- [`docs/production-decode-callgraph.md`](../docs/production-decode-callgraph.md) — live call graph
- `plans/parallel-sm-model.md` — wall model
- `plans/README.md` — which plans are active vs superseded
- `docs/dead-ends/` — refuted levers
- `docs/fulcrum-sota.md` — instrument semantics
