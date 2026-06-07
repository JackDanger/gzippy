# GATE 1 — FAST-BUT-REAL oracle pre-registered falsifier

Written BEFORE the run (orchestrator requirement). Branch reimplement-isa-l @ cbfb256.
T8, silesia-large, pure-rust-inflate, GZIPPY_FORCE_PARALLEL_SM=1, path=IsalParallelSM
asserted, interleaved best-of-N, sha-verified vs `gzip -dc`. Host-locked (no_turbo=1,
governor=performance) — verify in host log.

## What this run fixes vs the VOID 5b ceiling
5b's FLOOR_A was contaminated: a 656MB on-disk capture parse + 500MB CRC recompute
+ 41 ChunkData rebuilds, all IN-WALL. The decisive tell: floor **reused stayed at 12**
== baseline ⇒ "free decode" still delivered windows SLOWLY ⇒ the floor never tested
"prompt windows", which is the entire question. GATE 1 makes decode cost ≈0 while
windows remain REAL (byte-exact) AND are delivered PROMPTLY:
- RAM-resident replay (prebuilt-in-RAM, MOVED out on first request — NO per-call
  memcpy, NO per-call faults). The on-disk parse/prebuild must be EXCLUDED from the
  timed wall (warm the map before the timed loop, or amortize it across interleaved
  reps so it isn't attributed to the wall).
- CRC STRIPPED in the replayed chunk reconstruction (the 500MB CRC recompute at
  to_chunk_data:204-213 / sleep_replay:534-537 was a major in-wall confound; under
  decode≈0 the CRC dominates). Output bytes stay REAL (the captured `data` +
  `data_with_markers` are byte-exact), so sha against `gzip -dc` still passes — only
  the CRC32 *recompute work* is elided. (If sha requires the trailer CRC to match,
  gate trailer verification OFF for this oracle run and rely on sha of the bytes.)

## INSTRUMENT-VALIDITY GATE (must pass FIRST, before any verdict is read)
The instrument is VALID iff free-and-prompt decode actually delivers windows faster:
  **EAGER_PROBE_REUSED rises from 12 toward >= ~30 of ~39 chunks.**
- If reused stays ~12: the oracle STILL did not deliver fast windows. The instrument
  is NOT yet valid. DO NOT read a TIE/CAPPED verdict from it (do not trust a 5th
  broken instrument). Report "instrument invalid, reused=N≈12", diagnose why windows
  are still slow (likely: CRC/parse still in-wall, or prebuild not warmed, or the
  publish path itself is serial — which would then point GATE 2 at the lever), fix,
  re-run.

## VERDICT (only read once reused >= ~30 — instrument validated)
- **TIE-reachable** iff ALL of:
  - wall -> ~0.52-0.55s, AND
  - L_resolve median <= ~7.88ms, AND
  - window-absent stays ~90% (causal window_present; REJECT if it drifts toward 31% —
    that is clean-decoder divergence), AND
  - sha byte-exact vs `gzip -dc`.
  ⇒ window-wait was the cost; resolve stays cheap when windows arrive promptly ⇒ the
  INNER HUFFMAN DECODE loop is a clean arc to TIE (decode/window speed reaches TIE).
- **CAPPED** iff (despite reused >= ~30):
  - wall stays >= ~0.7s, OR
  - L_resolve median >= ~19.93ms.
  ⇒ genuinely window-wait-coupled / apply-compute-bound; inner loop alone does NOT
  reach TIE ⇒ the arc is the resolve/publish path (GATE 2's transliteration lever).
- **INCONCLUSIVE** iff reused >= ~30 AND L_resolve in (7.88, 19.93)ms AND wall in
  (0.55, 0.7)s. Report as such; do NOT pick a side. Decision routes to the advisor
  with the GATE 2 map.

## VOID conditions (STOP and report, no verdict)
- sha DIVERGES from `gzip -dc` (bytes wrong — a fast-but-WRONG floor is a loss).
- path != IsalParallelSM.
- host log lacks no_turbo=1 (frequency not pinned).
- replay hit% < ~90% (floor heavily re-running real decode).
- wall sd% > 5% (swap/bimodal — unreliable min).
- capture wrote 0 chunks, or the prebuild parse leaked into the timed wall (defeats
  the whole fix).

## Measured outputs to record (to /tmp/gate1-fastreal.txt, sentinel GATE1_DONE)
- baseline NORMAL wall (anchor — must reproduce ~0.917-0.929s ± spread)
- FAST-REAL floor wall (min, sd%)
- EAGER_PROBE_REUSED, EAGER_PROBE_SUBMITTED (the validity gate)
- replay hits/misses (hit%)
- L_resolve median (fulcrum model)
- dispatch_recv wall-critical
- causal window_present % (must stay ~90)
- sha result (OK/DIVERGE)
- host log no_turbo / governor
- rapidgzip anchor 0.5375s (reference)

## MEASURED RESULTS — RUN 1 @ bc139e9 (2026-06-06T19:33Z, host-locked no_turbo=1 governor=performance)
**VERDICT: INSTRUMENT INVALID — no TIE/CAPPED verdict readable (per the pre-registered validity gate).**
- GATE1_EAGER: submitted=34 **reused=3** (gate REQUIRED reused 12->>=30). Reused did NOT rise — it FELL below baseline 12. Windows arrived LESS promptly than the normal run.
- GATE1_FLOOR_PROBE: hits=41 misses=1, hit%=97.6, **sha=OK** (bytes byte-exact — CRC-strip is correct), probe_wall=**3.9730s** (same ~4s contaminated magnitude as VOID 5b's 3.667s).
- GATE1_CAPTURE: 42 chunks, **656512265 bytes** on /dev/shm (tmpfs). CRC-strip removed the recompute, but the per-process 656MB parse+prebuild is STILL in-wall.

### Why still invalid (CRC-strip necessary but insufficient)
Dominant remaining contaminant: the **656MB capture parse + full ChunkData prebuild, per process, inside the timed wall** — and it SERIALIZES window delivery (reused 12->3). Strong suspect: `prebuilt_map()` (decode_bypass.rs:588-600) builds the ENTIRE 42-chunk/656MB map on first access under ONE global Mutex; the first worker to call `replay()` blocks building all 656MB while every other worker serializes behind that Mutex -> windows delivered LATER than a normal decode, the opposite of "prompt." Structural flaw in the ORACLE, not the pipeline. 5th broken instrument (model.rs string-match; dead handoff_key; decode_bypass 5b; decode_bypass+CRC-strip 5c). Per CLAUDE.md rule 4 + this gate, DO NOT read a verdict.

### What a VALID oracle requires (re-attempt)
Decode~0 AND windows REAL AND windows PROMPT (reused must rise toward ~30):
- PRE-WARM the prebuild OUTSIDE the timed region (warmup pass), so the first timed worker hits an already-built map; OR build each chunk's ChunkData lazily PER-SLOT (not whole-map-under-one-Mutex).
- Avoid the 656MB materialization on the hot path: the windows that gate the wall are 32KiB end-windows; replay promptly what the consumer/window-map needs, keep payload memcpy lazy.
- Re-validate: reused >= ~30 BEFORE any wall/L_resolve verdict.
