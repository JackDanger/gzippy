# Parallel-efficiency map — silesia T4 isal (the named gap owner)

Status 2026-06-12. Banked reframe (advisor-gated, orchestrator-status 4a61d122):
gzippy decode CPU SUM 1497ms <= rg ~1781ms (0.84x) on silesia T4, but parallel
efficiency 81% vs rg ~92%; the ~64ms efficiency delta ~= the whole 67ms gap.
THE GAP IS WHEN WORK RUNS, NOT HOW FAST. This file is the pre-figured two-column
map (bias-guardrails T2: map before change) + the open dynamic questions the
staged Gantt instrument answers when neurotic returns.

## Static scheduling shape — VERIFIED VENDOR-PARITY (no lever here)

| Parameter | rapidgzip (vendor file:line) | gzippy | Verdict |
|---|---|---|---|
| Chunk size | 4_Mi default (ParallelGzipReader.hpp:280) | TARGET_COMPRESSED_CHUNK_BYTES = 4MiB (single_member.rs:44) | MATCH |
| Small-file shrink | size*2*P > file => ceil(file/(3P)) clamp 512Ki (PGR.hpp:295-305) | adjusted_chunk_size_bytes (single_member.rs:69-96, tested) | MATCH (ported) |
| silesia.gz @T4 | no shrink, ~17-18 chunks | 18 chunks observed | MATCH |
| Max decompressed/chunk | 20 x chunkSize (PGR.hpp:292) | max_decoded_chunk_size 20x (prod wiring; test fixtures mirror) | MATCH |
| Prefetch ramp | Prefetcher.hpp:88-225 FetchNextAdaptive | prefetcher.rs cited port (:81 cites Prefetcher.hpp:88-225) | MATCH (cited port) |
| Hit-path prefetch drive | BlockFetcher.hpp:297-309 | chunk_fetcher.rs:1525-1540 (kill-switch A/B: SLACK) | MATCH + causally null |
| apply_window priority | submit(-1) (BlockFetcher.hpp:608-611) | submit(-1) (chunk_fetcher.rs:2445,2455) | MATCH (advisor-verified) |
| Window publish thread | consumer (GzipChunkFetcher.hpp:570-574) | consumer 61/0, 51/0 (TID counters) | MATCH (measured) |
| ISA-L call granularity | 128KiB staging+output (isal.hpp:205-207) | inflate_wrapper.rs:31-43 (advisor-verified) | MATCH |

Conclusion: every static parameter audited so far is vendor-identical. The 81%-vs-92%
delta must be DYNAMIC — the Gantt instrument (probe/parallel-eff 27b1ffcf,
GZIPPY_CHUNK_PHASE=1) decides between:

## Open dynamic questions (Gantt answers; do NOT pre-judge)

1. STRAGGLER GEOMETRY: where does the 174ms marker-only tail chunk run? Started
   last and gating the wall (quantization) vs started early (something else binds).
   rg faces the same 18-chunk geometry — does its tail cost 174ms too (its tail
   would ALSO be window-absent... unless its window for the tail publishes earlier)?
2. POOL-FILL RAMP: per-worker first-task-start times — is there a startup gap
   (first P submits serialized behind the block finder?) that rg doesn't pay?
3. IDLE GAPS MID-RUN: worker idle between tasks (waiting for prefetch dispatch /
   window-ready) — count and place them; cross-ref confirmed_offset_prefetch_gap.
4. WINDOW-READY TIMING: chunks decode marker-mode iff predecessor window unpublished
   at THEIR start — per-chunk window-absent flag vs start time vs predecessor finish:
   is gzippy's marker fraction higher than rg's at the same T (the decode-twice tax
   is CPU-cheap [body slack] but apply_window + markers-replace adds latency on the
   chunk's CRITICAL path even when CPU-slack)?
5. RESIDUAL SUBPHASES: the 247ms (f) bucket (CRC32-over-ISA-L, boundary replay,
   finalize, setup, reserve) — instrumented into 5 AtomicU64s, same probe.

## Resume sequence (when neurotic answers)

Worktree /tmp/gz-peff branch probe/parallel-eff (27b1ffcf) has everything; the
worker report (task ae3e60aaa/a54c... chain) contains the exact build+run commands.
Order: Task-1 CPU gate confirm (fresh rg --verbose same-session; semantics now
known: thread-summed CPU; correct gzippy numerator by adding applyWindow+CRC) ->
Gantt 3 traced runs -> residual table -> pre-registered sleep perturbation of the
named candidate. Then `fulcrum locate` (now real: park-aware residual, v0.3.0+
local) on the same traces as the cross-check instrument.
