# CAMPAIGN CHARTER — gzippy → rapidgzip parity (owner's constitution)

You (the OWNER agent) fully own this campaign. This doc is your constitution; keep it current
as the single source of truth. Supervisor = thin relay/cleanup only.

## GOALS (crisp)
Two flag-gated parallel single-member gzip decode paths, both FAITHFUL ports of what rapidgzip
ACTUALLY does (faithfulness is defined by rapidgzip's CODE, never by any memory line):
1. **gzippy-native** (default): does literally what rapidgzip does, **entirely in Rust, no
   C-FFI** — including using **u8 wherever rapidgzip uses u8, full stop**. Inline ASM is allowed.
   Target: a **1.0× wall TIE** with rapidgzip. Nothing less is accepted.
2. **gzippy-faithful (isal)**: the same, but hands the clean tail to **ISA-L via C-FFI** (the
   reference/comparison baseline, = rapidgzip's WITH_ISAL build).
Both use u8 in the clean tail; they differ ONLY in whether the u8 decoder is gzippy-Rust or
ISA-L-C. Done = gzippy-native ties rapidgzip on the locked whole-system wall across the workload
matrix, byte-exact, with the structure faithfully mirroring rapidgzip.

## PROCESS (the method — read this twice)
**We do NOT hunt individual levers.** A lever list is how you climb to a local optimum and stall.
Instead we treat the decoder as ONE system and repeatedly do exactly this loop:

1. **Perceive the whole system** with whole-system numbers: the real end-to-end interleaved
   wall (sha-verified, locked harness), at the thread counts that matter. This — not any
   component's busy-time, rate, or latency-share — is the only truth. Producer-side attribution
   is analyst-biasable and has manufactured phantom levers all campaign (rate-ratio "plateaus,"
   busy-time "blame"). The whole-system wall is the verdict.
2. **Find what is CURRENTLY the bottleneck** — the one thing that, right now, sets that wall.
   Establish it CAUSALLY: perturb a candidate and watch the whole-system wall respond
   (monotonic ⇒ on the critical path; flat ⇒ slack), with a frequency-neutral control. Never
   conclude a bottleneck from attribution alone.
3. **Fix that bottleneck** — whatever it happens to be (engine arithmetic, a production-path
   overhead, scheduling, memory, window handling). We don't care which; we fix the binder.
   Rewrite it CORRECTLY (our cost is dominated by shortcuts, not by correct rewrites). Byte-exact.
4. **Re-perceive the whole system.** Fixing the bottleneck MOVES it somewhere else. Go to 1.
   Stop when the whole-system wall ties rapidgzip.

This is bottleneck-following on whole-system numbers, not lever-shopping. A "win" is only real if
it moves the whole-system wall; a byte-exact change that ties is still KEPT (rule 7a) because it's
correct and its gain may be latent behind the current binder — but it does not count as progress
toward the tie until the wall moves.

## NON-NEGOTIABLE DISCIPLINES
- Byte-exact ALWAYS: dual-sha 028bd002c89c9a909ccdbc2af0a223de285348edb014ccc8e27d297f52cb410f on
  BOTH features; all lib tests + the adversarial seam test green. Wrong bytes = void.
- Numbers ONLY from the locked guest harness (interleaved, sha-verified). Never a hand script.
- Independent disproof advisor (synchronous) corroborates every consequential claim BEFORE you
  rely on it. Pre-register a falsifier before each experiment; bound a speed-up CEILING by
  removal/oracle (a slow-down slope proves "on the path," never the payoff).
- No memory or plan line may compete with the goal or redefine faithfulness away from rapidgzip's
  actual code. If one does, correct it.
- Escalate to the supervisor/user ONLY on a genuine FORK that trades off a user constraint
  (1.0× bar vs no-FFI vs faithfulness) — e.g. if a bottleneck proves unfixable in pure-Rust+ASM.
  Otherwise drive with agency.

## HOW YOU DELEGATE (carefully)
You own the work; you delegate via your OWN `claude -p --model opus --permission-mode
bypassPermissions` subagents. Rules that have cost whole turns:
- Run subagents SYNCHRONOUSLY (block with `timeout`, collect in-turn). There is NO auto-reinvoke —
  do NOT background a subagent and yield for a "notification"; you will simply die and the
  supervisor must re-drive. Run measurements YOURSELF (Bash holding the ssh); delegate research
  and the disproof advisor as synchronous calls.
- NO detached `sleep` leader-lock sentinel (it orphans). Leave NO orphaned processes — before you
  finish, pgrep must show none of your claude -p / sleep children.
- SOURCE-VERIFY any premise first-hand before acting on it (a wrong premise — "gzippy never
  re-targets," the "window-discard" — has burned turns). Serialize builds via cargo-lock.sh.
- Keep THIS charter + plans/orchestrator-status.md current so a fresh owner-spawn can resume.

## CURRENT STATE (2026-06-07, owner turn, branch reimplement-isa-l, HEAD f1aceee1) — **COUNTER RENAMED (anti-inversion) + SCHEDULING/SERIAL CEILING BOUNDED via real oracles. ADVISOR REFUTED my arithmetic F2 over-reach.** The T8 TIE IS reachable (seedfull oracle = 1.029× T8, 1.121× T16 WIN, sha-exact) BUT scheduling-overlap AND the window-absent marker-engine rate are LIVE + ARCHITECTURALLY COUPLED terms — neither is cleanly isolable (window-present ⇒ clean engine, gzip_chunk.rs:790). **The prompt's "engine slack-masked, binder is scheduling" premise is NOT confirmed AND my counter-arithmetic "engine binds (F2)" is REFUTED** (Rule-3 extrapolation; the sum was a strict upper bound). Status = F3/both-live. **SUPERVISOR GATE — ceiling bounded, NO engine fix landed (binder coupled/unconfirmed); next loop must CAUSALLY perturb before any work-stretch.**

### THIS TURN — renamed the inversion-prone counter (byte-exact), then BOUNDED the scheduling/serial ceiling with REAL removal oracles on the locked guest. Advisor (synchronous, read-only) REFUTED my arithmetic over-reach; resolution is oracle-grounded.
- **COUNTER RENAME (commit f1aceee1, byte-transparent instrumentation):** `BOOTSTRAP_POST_FLIP_U16_BYTES` → `BOOTSTRAP_CLEAN_FLIPPED_BYTES` (gzip_chunk.rs:97/:1491 + chunk_fetcher.rs:947 + GZIPPY_VERBOSE label). It counts output bytes of bootstrap blocks that ended CLEAN (`!contains_marker_bytes`) — the marker-FREE COMPLEMENT, NOT "bytes decoded into the u16 marker ring after the flip" as the old name+doc claimed. It had been read backwards repeatedly (the exact C3 counter-inversion the prior advisor refuted). New label now self-documents: `clean_flipped_bytes=1425448 (2.0% of body = marker-FREE complement; marker loop owns the other 98.0%)`. Compiles clean; faithful_u8_flip_seam test green.
- **FIRST-HAND VERBOSE (locked guest REDACTED_IP double-ssh, 16c gov=perf, gzippy-isal native synced to HEAD f1aceee1, sha 028bd002…cb410f every cell, T8):**
    | metric | gzippy HEAD | rapidgzip | ratio |
    | decodeBlock SUM | 0.803s | 0.502s | 1.60× |
    | Theoretical Optimal (÷T) | 0.100s | 0.068s | 1.47× |
    | Total Real Decode | 0.116s | 0.084s | 1.38× |
    | std::future::get | 0.089s (T16: 0.046s) | 0.064s | 1.39× |
    | serial tail (wall−RealDecode) | 0.058s | ~0.043s | 1.35× |
    | WALL (interleaved best, measure.sh) | 0.174-0.177s | 0.130s | **0.736-0.755×** |
  Note Real Decode 0.116 is BELOW rg's WALL 0.130 (the prompt's cached 0.137 was pre-mergefix/stale). T16: HEAD 0.162s = **0.885×** (rg slows to 0.144 at T16 on 17 chunks).
- **REMOVAL ORACLE #1 — seedfull (GZIPPY_SEED_WINDOWS, sha-exact):** all 17 chunks window-seeded ⇒ CLEAN engine, 0 spec-failures, Fill 90%. T8 wall **0.128s = 1.029× rg = TIE**; T16 **0.128s = 1.121× rg = WIN**. seedfull's future::get 0.083s ≈ HEAD's 0.089s. **This IS the faithful "perfect window-overlap" oracle** — the ONLY way to give the consumer pre-resolved windows (remove head-of-line wait) is to seed windows, which ALSO flips the engine clean (coupling, gzip_chunk.rs:790 vs :826). A pure-scheduling oracle keeping the marker engine is IMPOSSIBLE in-architecture.
- **NEGATIVE CONTROL — GZIPPY_NO_PREFETCH (sha-exact):** T8 wall **0.523s = 0.253× rg (3× SLOWER)**. Removing the prefetch overlap is catastrophic ⇒ scheduling is FIRMLY on the critical path. + future::get HALVES T8→T16 (0.089→0.046) = signature of CRITICALITY (slack does not scale with cores).
- **INDEPENDENT DISPROOF ADVISOR (synchronous, read-only, plans/scheduling-ceiling-advisor-verdict.md):** C1 (engine reaches wall) UPHELD-WITH-CAVEATS (the HEAD→seedfull A/B moving 0.040s on the non-future::get axis is the real evidence, NOT the banned attribution ratios — but "reaches the wall" ≠ "is THE binder"). **C2 (scheduling not the binder) REFUTED** — I read future::get's halving backwards; it IS a criticality signature; NO_PREFETCH 3× regression confirms it. **C3 (arithmetic F2 ceiling = loss) REFUTED (load-bearing)** — 0.116+0.043 is my OWN strict upper bound (double-counts the overlapping tail); the only LOWER bound (decode-phase wall 0.116 = 0.89× rg) is in TIE territory; concluding F2 from a hand sum with no oracle violates Rule 3. **C4 (next binder = backward marker scan emit_backref_ring::<true> :3006-3027) UPHELD-WITH-CAVEATS → effectively UNCONFIRMED** — the scan is fast-path-skipped once `distance_marker>=distance` (:3002) and the isal build FLIPS to clean u8 at 32KiB (gzip_chunk.rs:949), confining it to the per-chunk bootstrap (<1% of a multi-MB chunk); implausible as the prime 1.6× term; needs a causal perturbation.
- **BOUNDED CEILING (honest, oracle-grounded): the T8 TIE IS reachable** (seedfull proves it, F1) but BOTH the scheduling overlap AND the window-absent marker-engine rate are LIVE, COUPLED terms; neither is isolable in gzippy's architecture (window-present ⇒ clean). **rg ties UNSEEDED at the same 34.5% markers because its marker engine is fast (decodeBlock 0.502 vs 0.803 = 1.6×)** — so the faithful path is to make the window-absent decode cheaper at the wall.
- **rg's MECHANISM (source-verified, vendor GzipChunkFetcher.hpp):** `waitForReplacedMarkers` (:479) queues the head chunk's marker-replace, then USES THE WAIT to harvest ready futures + `queuePrefetchedChunkPostProcessing` (:513, full sorted prefetch-cache scan, queue post-process for every chunk whose predecessor window is available). The LAST window is inserted by the MAIN thread (:559-561, *"the critical path that cannot be parallelized... do not compress the last window to save time"*) — rg explicitly names window-publish as THE serial critical path and minimizes it. gzippy ALREADY ports this (queue_prefetched_marker_postprocess chunk_fetcher.rs:1592/1702 + prefetch pump during wait). So the consumer STRUCTURE is faithfully ported; the residual is dispatch TIMING (windows published slightly later ⇒ more chunks window-absent at high T), NOT a missing mechanism and NOT horizon DEPTH (vendor-identical).
- **SCOPED FIX for the NEXT loop (do NOT start — supervisor gate; must CAUSALLY PERTURB FIRST per advisor C4):** TWO coupled faithful candidates, each needs a confirming perturbation before a work-stretch: (a) faster window-absent u16 MARKER engine (the 1.6× decodeBlock gap reaching the wall via Real Decode 1.38×) — but C4's specific "backward marker scan" hypothesis is UNCONFIRMED/implausible (flip-to-clean at 32KiB confines it); the real remaining engine term is more likely the post-flip u8 CLEAN rate + u16 bootstrap traffic, which a SLOW-INJECT/oracle perturbation must locate; OR (b) publish predecessor windows EARLIER so more chunks hit the clean path at high T (closing project_confirmed_offset_prefetch_gap dispatch-TIMING). Bound each with a REMOVAL oracle; never the slow-down slope.
- **GUEST STATE:** /root/gzippy/src rsynced to HEAD f1aceee1 (gzippy-isal native build /tmp/gzbuild-head, sha 028bd002…cb410f). Drivers /tmp/head_measure.sh, /tmp/seedfull_measure.sh, /tmp/t16_measure.sh, /tmp/noprefetch_measure.sh, /tmp/head_verbose.sh, /tmp/seedfull_verbose.sh, /tmp/rg_verbose.sh (all `bash`). Seeds /tmp/seeds.bin. NO orphan processes (advisor wrapper + sleep killed; guest pgrep clean).

## SUPERSEDED — PRIOR CURRENT STATE (2026-06-07, owner turn, HEAD 04fda86d) — **PORT (i) LANDED: rg's multi-cached u16 marker FAST LOOP. Byte-exact. T8 wall = TIE (no move), KEPT per 7a. Advisor C1 UPHELD, C3 REFUTED my mechanism.** The marker decode SUM gap (decodeBlock ~1.9× rg) is SLACK-MASKED at the wall (Fill 87%, wall unchanged). Whole-system T8 wall STILL ~0.73× rg. NEXT = the binder is NOT the marker compute — re-perceive: the engine SUM is ~1.9× but slack-masked (matches Phase-0 oracle TIE-vs-TIE). **SUPERVISOR GATE — marker loop measured + committed (TIE, kept); next binder NOT yet located.**

### THIS TURN — ported rg's multi-cached u16 marker fast loop, byte-exact, remove-and-measure on the locked guest. Result: a faithful TIE (kept), and an important PREMISE CORRECTION.
- **THE CHANGE (commit 04fda86d, faithful port of vendor `readInternalCompressedMultiCached` deflate.hpp:1585-1666):** added a speculative software-pipelined FAST LOOP to the u16 MARKER path (`read_internal_compressed_specialized::<true>`, marker_inflate.rs new `'mfast` loop), mirroring the clean path's existing fast loop. rg runs the SAME tight multi-cached loop for u16 markers as for u8 clean (templated on `Window`, no separate slow marker path); gzippy's clean path already had its fast loop but the MARKER path was stuck on the careful per-symbol loop. Three faithful u16 deltas: (1) literal store widened to u16 via an 8-byte speculative store `(p&0xFF)|((p&0xFF00)<<8)|((p&0xFF0000)<<16)`, value-identical to the careful loop's `write(code&0xFF)`; (2) `distance_marker += lit_prefix` per packet, back-refs via the SAME `emit_backref_ring::<true>` (marker scan maintained inside); (3) no `distance>decoded+emitted` range check (vendor const-folds it for marker windows).
- **BYTE-EXACT:** gzippy-native arm64 (T1/T8/T16) + gzippy-isal guest x86_64 (T1/T8/T16) BOTH sha 028bd002…cb410f via path=ParallelSM. 856 lib tests pass (1 fail = pre-existing flaky `diff_ratio` timing micro-test). Adversarial seam test (`faithful_u8_flip_seam_max_distance_backref_vs_flate2`) + native_fold_parity green.
- **REMOVE-AND-MEASURE (locked guest REDACTED_IP double-ssh, 16c gov=perf turbo-on, taskset 0,2,4,6,8,10,12,14, T8, measure.sh interleaved N=11, RAW=68229982, sha-OK every run): markerfast vs mergefix(prior HEAD 77a02f5f) = +1.2% / +3.0% / +0.0% across 3 interleaved runs = TIE** (within 10-38% spread; per charter "Δ < spread ⇒ TIE"). Per-stage trace: decodeBlock 0.9568→0.9485s (~0.9%), body_rate 203→207 MB/s. rg decodeBlock 0.500s ⇒ gzippy still ~1.9×, but Total Real Decode 0.137s / Fill 87% / wall 0.175s = the engine SUM is SLACK-MASKED (matches Phase-0 TIE-vs-TIE). **VERDICT: faithful TIE — KEPT per rule 7a; gain latent behind the current (non-engine) binder.**
- **ADVISOR (plans/marker-loop-port-advisor-verdict.md, synchronous read-only, source-verified):** C1 BYTE-EXACT+FAITHFUL **UPHELD** (widening correct b0/b1/b2/0; no ring-wrap straddle; no bit-cursor desync; dropped range-check faithful; emit_backref_ring is literally the same fn both loops call). C2 WALL-TIE **UPHELD-WITH-CAVEATS** (honest no-regression + interleaved is freq-neutral, but 10-38% spread can't detect ≤10-20% — nearly uninformative). C3 my "marker path is only ~2% of body ⇒ TIE expected" mechanism **REFUTED (load-bearing):** I read `BOOTSTRAP_POST_FLIP_U16_BYTES` BACKWARDS — it increments only when a block ends CLEAN (`flipped_clean = !contains_marker_bytes()`, gzip_chunk.rs:1489-1495), so 2.0% is the CLEAN sliver the loop does NOT touch; the loop's actual domain is the COMPLEMENT ~98% of bootstrap body. (The exact counter-inversion the charter's u16-ceiling correction warns about — I repeated it.) ⇒ the TIE is NOT a small-domain ceiling. Commit message corrected to strike the inverted rationale.
- **CORRECTED PREMISE for the next loop:** the "decodeBlock 1.69× = the marker loop" attribution is now suspect — the marker fast loop owns ~98% of bootstrap body yet barely moved decodeBlock (0.9%) and did NOT move the wall. The engine SUM gap (~1.9×) is real but SLACK-MASKED at Fill 87% (Phase-0 already showed engine TIE-vs-TIE when seeded). **The T8 binder is NOT the per-thread engine compute.** Re-perceive: the wall is 0.73× rg with Fill 87% + Total Real Decode 0.137s ≈ rg's whole wall — the gap is the SCHEDULING/SERIAL term (pool-fill + in-order consumer head-of-line wait), the long-deferred project_confirmed_offset_prefetch_gap binder. NEXT loop should bound THAT with a removal oracle, not chase the slack-masked engine further.
- **GUEST STATE:** /root/gzippy tree RESTORED to baseline (marker patch reversed, marker_inflate.rs sha 7b87c5bd) + the mergefix overlay still applied (chunk_data.rs/chunk_fetcher.rs). Builds: /tmp/gzbuild-base + /tmp/gzbuild-mergefix (prior) + /tmp/gzbuild-markerfast (THIS turn, gzippy-isal native, sha 028bd002…cb410f). Drivers /tmp/markerfast_measure.sh + /tmp/markerfast_trace.sh + /tmp/sha_markerfast.sh (use `bash`). Patch /tmp/marker_fastloop.patch. No orphan processes (advisor wrapper + sleep killed; guest clean).


### THIS TURN — landed the merge-removal (cheapest+most-uncertain of the two ports, measured FIRST per advisor), byte-exact, remove-and-measure on the locked guest.
- **THE CHANGE (faithful port of vendor `applyWindow` swap+views, DecodedData.hpp:325-390 = narrow → swap → VectorViews → `dataWithMarkers.clear()`, NO output-size copy):** `resolve_chunk_markers_on_chunk` (chunk_fetcher.rs:2453) now DROPS `merge_resolved_markers_into_data()` (the redundant ~68MB full-output memcpy — `prepend_narrowed_from_markers`, segmented_buffer.rs:356-378, allocates `n+data.len()` and `extend_from_slice`s the WHOLE clean payload too) AND the eager `recycle_markers_after_resolution()`. The narrowed marker bytes STAY in `data_with_markers` (u8 view of the u16 backing) with `narrowed_len` set; the consumer emits them zero-copy via `append_output_iovecs`→`append_narrowed_iovecs` (chunk_data.rs:1609, already supported narrowed_len>0). Marker-segment recycle DEFERRED behind the consumer writev via the existing `defer_chunk_recycle`→`recycle_decoded_buffers` (frees BOTH data + data_with_markers). `contains_markers` (chunk_data.rs:577) now treats `narrowed_len>0` as resolved (post-narrow the u16 high bytes are stale so `all_resolved()` would misread → `has_been_post_processed` depends on this). `populate_subchunk_windows` assert relaxed (copy_window_at_chunk_offset already branches on narrowed_len>0 at :1220). + a debug-only double-resolve tripwire in `resolve_and_narrow_markers_in_place` (advisor rec; byte-transparent). New test `populate_subchunk_windows_unmerged_view_based_apply_window` locks the un-merged path.
- **BYTE-EXACT:** gzippy-isal native (guest x86_64) + gzippy-native (local arm64) BOTH sha 028bd002…cb410f at T1 AND T8 via path=ParallelSM. 856 lib tests pass (the 1 fail = pre-existing flaky `diff_ratio` timing micro-test, fails IDENTICALLY on unmodified 507d6ecb — confirmed by stash test). Adversarial seam test + native_fold_parity green. New un-merged test green (debug build exercises the tripwire, doesn't fire).
- **REMOVE-AND-MEASURE (NOT the SUM, per advisor Q4): locked guest REDACTED_IP double-ssh, 16c gov=performance turbo-on, taskset 0,2,4,6,8,10,12,14, T8, measure.sh interleaved N=11, RAW=68229982, sha-verified=OK every run. base(WITH merge) vs mergefix(REMOVED), both gzippy-isal native target-cpu=native:**
    | run (load) | base | mergefix | mergefix Δ | base vs rg | mergefix vs rg |
    | run1 (1.64) | 0.2291s | 0.2045s | **+12.0%** | 0.624× | 0.699× |
    | run2 (2.80) | 0.2128s | 0.1900s | **+12.0%** | 0.684× | 0.766× |
    | run3 (1.86, cleanest 6-13% spread) | 0.2006s | 0.1765s | **+13.7%** | 0.651× | 0.739× |
  Sign STABLE across 3 interleaved runs; load-invariant (delta holds at 1.64/2.80/1.86) ⇒ NOT a turbo/frequency artifact. Interleaved measurement is freq-neutral by construction (both tools alternate trials per N). **VERDICT: merge-removal moves the T8 wall ~12% (rg ratio 0.65×→0.73×). KEEP.** Mechanism (advisor Q4): the per-chunk O(whole-chunk) alloc+memcpy landed on the consumer's blocking recv for un-pre-resolved head-of-line marker chunks; removing it un-blocks that critical fraction.
- **INDEPENDENT DISPROOF ADVISOR (synchronous, read-only, plans/merge-removal-advisor-verdict.md):** C1 BYTE-EXACT UPHELD (vendor citation accurate; change is MORE faithful, not a divergence; contains_markers narrowed_len guard required+correct). C2 WALL UPHELD-WITH-CAVEATS (memcpy IS literally the whole clean payload; +12% plausible; stable+sha-identical rules out turbo/wrong-fast; caveat: alloc+copy removed together, attribution not isolated but remove-and-measure was the right method). C3 CORRECTNESS UPHELD-WITH-CAVEATS (no use-after-recycle on any of 3 emit paths — pipe boxes chunk covering BOTH buffers, non-pipe + buffered are sync-then-defer; re-resolution gates hold via !markers_resolved). Single correction ADOPTED: added the double-resolve debug tripwire (the merge used to empty the buffer as a guard; now safety rests on markers_resolved — tripwire restores defense-in-depth).
- **NEW WHOLE-SYSTEM WALL vs rapidgzip: T8 ~0.73× (was ~0.65×).** Still a LOSS — the remaining gap is port (i): rg's multi-cached u16 marker loop (decodeBlock 1.69×), the larger of the two divergences.
- **SCOPED NEXT (do NOT start — supervisor gate): port (i) rg's multi-cached u16 marker loop** to close decodeBlock 1.69× (vendor readInternalCompressedMultiCached deflate.hpp:1453, ONE loop over the u16 window, constexpr-gated marker arms). Larger change; advisor-gated; remove-and-measure. Re-check the gather/crc ~1.5× residual (advisor Q5 third term) after.
- **GUEST STATE:** /root/gzippy reset to clean 507d6ecb source (prior overlays git-stashed as `owner-overlays-507turn`) + this turn's merge-removal applied via /tmp/mergefix.patch. Builds: /tmp/gzbuild-base (507d6ecb WITH merge) + /tmp/gzbuild-mergefix (merge removed), both gzippy-isal native, both sha 028bd002…cb410f. Drivers /tmp/merge_measure.sh + /tmp/sha_check.sh (use `bash`). No orphan processes.

## SUPERSEDED — PRIOR CURRENT STATE (2026-06-07, HEAD 507d6ecb +substep-timers-on-guest) — CEILING BOUNDED → **T8 TIE needs TWO faithful ports, not one: (i) rg's multi-cached u16 marker loop (decodeBlock 1.69×) + (ii) rg's view-based applyWindow that skips a redundant full-output memcpy (the `merge` step, 0.12-0.13s SUM = the apply_window divergence)**. apply_window is NOT at parity — but the excess is a removable copy faithful-to-rg, NOT the LUT gather (which is ~1.5-2× and algorithmically identical). rg's "applying the last window" = **0.032s** (NOT the charter's cached 0.113s — that number was WRONG). Advisor: all findings UPHELD-WITH-CAVEATS, none refuted. **SUPERVISOR GATE — do NOT start the fix build (ceiling now bounded; report + gate).**

### THIS TURN — paid the OWED apply_window measurement + source-verified rg's marker-decode mechanism FIRST-HAND, then DECOMPOSED gzippy's apply_window.
- **rg's MARKER decode mechanism (source-verified, vendor deflate.hpp):** `readInternal` (:1428) dispatches by Huffman-coding TYPE not marker-vs-clean; with WITH_ISAL the lit/len path is `readInternalCompressedMultiCached` (:1453) for BOTH u16 markers AND u8 clean (templated on `Window`). It is ONE loop; `containsMarkerBytes` is a constexpr from the element type (:1600). Marker-vs-clean differ ONLY in cheap constexpr-gated arms: m_distanceToLastMarkerByte counter (:1311-1317), post-memcpy back-scan (:1379-1389), inverse window-range-check skip (:1652-1655). resolveBackreference fast arm is `std::memcpy` for BOTH (:1376). ⇒ rg's marker decode is fast because it runs the SAME multi-cached fast loop on the u16 window — there is NO separate slow marker path in rg. The faithful target = port rg's multi-cached u16 loop (NOT bolt AVX onto gzippy's loop — that's the E234 0.41× plateau). Caveat (advisor Q1): markers are u16 by construction ⇒ a faithful port is ~2× the mem traffic of the u8 clean path; promise "marker == rg's u16 multi-cached loop," NOT "marker == u8-clean speed."
- **OWED apply_window measurement (locked guest REDACTED_IP double-ssh, 16c gov=perf turbo-on load ~1.0, taskset -c 0,2,4,6,8,10,12,14, T8, RAW=68229982, sha 028bd002…cb410f EVERY run, /tmp/gzbuild-isal gzippy-isal native, measurement-only sub-step timers added byte-exact (NOT committed), 3 runs):**
    | term (SUM across 15 marker chunks) | gzippy | rg --verbose (first-hand) | ratio |
    | decodeBlock | 0.838s | 0.497s | **1.69×** |
    | gather (LUT resolve+narrow = rg's applyWindow analogue) | 0.044-0.064s | "applying the last window" **0.032s** | ~1.5-2× (algo IDENTICAL) |
    | crc (update_narrowed_crc) | 0.013-0.019s | "checksum" 0.0096s | ~1.5× |
    | **merge_resolved_markers_into_data** | **0.116-0.134s** | std::swap (~0s) | **structural divergence** |
    | subwin (populate_subchunk_windows) | 0.010-0.012s | window export (separate) | — |
    | TOTAL apply_window_us | 0.19-0.27s | — | — |
- **THE apply_window DIVERGENCE = `merge` (chunk_data.rs:1589 → segmented_buffer.rs:356 `prepend_narrowed_from_markers`):** allocates a fresh n-byte buf and `extend_from_slice` COPIES every narrowed byte (the whole ~68MB output) into `data`. rg does NOT do this — DecodedData.hpp:368 `std::swap` + VectorViews into the marker buffers in place (:371-388), no output-size copy. gzippy ALREADY HAS the zero-copy emit (`append_output_iovecs`/`append_narrowed_iovecs`, chunk_data.rs:1609 / segmented_markers.rs:532) ⇒ the merge-copy is REDUNDANT for the iovec writer. The LUT gather is FAITHFUL+identical to rg (`base[i]=lut[v]` ↔ rg `target[i]=fullWindow[chunk[i]]`, DecodedData.hpp:335-337) — the gap is the copy, not the algorithm.
- **INDEPENDENT DISPROOF ADVISOR (synchronous read-only, plans/marker-kernel-ceiling-advisor-verdict.md):** all findings UPHELD-WITH-CAVEATS, NONE refuted. Q1 marker==multi-cached-loop fair (caveat: u16 ⇒ ~2× clean traffic). Q2 apply_window apples-to-apples (caveat: rg's 0.032s already includes its swap+views; honest framing = rg 0.032 vs gzippy gather+merge). Q3 merge IS removable byte-exactly + faithful-to-rg (every consumer — writer, window-extraction, CRC, data_prefix_len>0 — already supports the un-merged state, traced) BUT it is a STRUCTURED change not a delete: must defer marker-recycle behind the consumer writev (else use-after-recycle), relax the populate_subchunk_windows `narrowed_len==0` assert (chunk_data.rs:1291), keep narrowed_len set through write. Q4 (LOAD-BEARING) **do NOT trust −0.12s SUM as the wall delta** — merge runs on the pool; its wall cost is only the un-overlapped fraction landing on the consumer's `recv_post_process_blocking` (chunk_fetcher.rs:1769) for un-pre-resolved head-of-line marker chunks, bounded by resolve-ahead hit rate (project_confirmed_offset_prefetch_gap). Provable ONLY by remove-and-measure (freq-neutral control), never the SUM. Q5 ceiling DIRECTIONALLY SOUND, two ports are the right faithful levers, NOT yet a proven TIE.
- **BOUNDED CEILING (REVISED, honest): T8 TIE plausibly reachable in PURE-RUST via TWO faithful ports — NOT one.** (i) rg's multi-cached u16 marker loop (closes decodeBlock 1.69×); (ii) rg's view-based applyWindow = drop the redundant `merge` memcpy, emit narrowed-marker iovecs (closes the 0.12-0.13s `merge` divergence). PLUS a smaller third residual the advisor flags (gather ~1.5-2× + crc ~1.5× = SegmentedU16 multi-segment walk + per-chunk LUT rebuild vs rg's contiguous chunk + hoisted fullWindow; may need (iii) hoist the LUT build / contiguous narrow target). The prior "marker-COMPUTE only" ceiling was OPTIMISTIC exactly as advisor Q4 warned — apply_window is a real second term and is NOT at parity.
- **SCOPED FIX FOR NEXT LOOP (do NOT start — supervisor gate): land merge-removal FIRST** (cheapest of the two ports, payoff most uncertain ⇒ measure first per advisor): convert `merge_resolved_markers_into_data` to rg's swap+views model — skip `prepend_narrowed_from_markers`, keep narrowed bytes in the marker pages, emit via `append_narrowed_iovecs`, DEFER marker-recycle behind the consumer writev, relax the subchunk `narrowed_len==0` assert. Byte-exact + measure the interleaved T8 wall (freq-neutral control). THEN the multi-cached u16 marker loop (decodeBlock). Each advisor-gated, each remove-and-measure (never the SUM, never the slow-down slope).
- **GUEST STATE:** /root/gzippy src @7bf26096 + oracle overlay + decompose knobs + THIS turn's measurement-only sub-step timers in chunk_fetcher.rs (gather/crc/merge/subwin, applied via /tmp/patch_resolve.py + /tmp/patch_merge.py on guest, NOT committed locally — byte-exact, sha unchanged). Build /tmp/gzbuild-isal (gzippy-isal native, rebuilt this turn). Drivers /tmp/applywin_measure.sh + /tmp/substep2_measure.sh (use `bash`). Seeds /tmp/seeds.bin. No orphan processes.

## SUPERSEDED — PRIOR CURRENT STATE (2026-06-07, HEAD 5e9905c8 +decompose-knobs) — BUNDLE DECOMPOSED → **THE T8 SUB-LEVER IS marker-COMPUTE: gzippy's window-absent u16 marker decode is ~2× SLOWER per byte than rapidgzip's** (advisor UPHELD-WITH-CAVEATS). Boundary-alignment + spec-failures are NOT the cost. CEILING = ≤ T8 1.0× TIE, conditional on applyWindow parity. **SUPERVISOR GATE — do NOT start the fix build (bound-ceiling-first; one owed measurement remains).**

### THIS TURN — DECOMPOSED the GZIPPY_SEED_WINDOWS bundle (advisor's 3-removal confound: a=marker-compute, b=boundary-alignment, c=spec-failure re-decodes). Added 2 measurement-only env knobs (OFF==identity, byte-exact, NOT committed): `GZIPPY_SEED_NO_WINDOWS=1` (suppress seeded-window fallback ⇒ seed-only-boundaries) + `GZIPPY_SEED_NO_BOUNDARIES=1` (skip block_finder pre-seed ⇒ seed-only-windows). seed_windows.rs + chunk_fetcher.rs.
- **MEASURED (locked guest REDACTED_IP double-ssh, 16c gov=perf turbo-on load 1.3-2.0, measure.sh interleaved N=11 CPUS=0,2,4,6,8,10,12,14 RAW=68229982 sha-OK=028bd002…cb410f every cell, 2 runs):**
    | cell | what's seeded | wall | vs rg |
    | rg (rapidgzip 0.16.0)              | —          | 0.132s | 1.000 |
    | seedfull (windows+boundaries)      | both       | 0.126-0.134s | **~1.00× TIE** |
    | onlywin (NO_BOUNDARIES, windows)   | windows    | 0.199s | 0.66× LOSS |
    | onlybnd (NO_WINDOWS, boundaries)   | boundaries | 0.198-0.205s | 0.66× LOSS |
    | prod (no seeding)                  | nothing    | 0.198-0.203s | 0.66× LOSS |
  **onlywin ≈ onlybnd ≈ prod (Δ<spread); only seedfull (BOTH) ties.** Pre-reg formula: f_windows≈0, f_boundary≈0, yet seedfull ties ⇒ SUPER-ADDITIVE/COUPLED (pre-reg branch-4).
- **MECHANISM (per-cell counters, GZIPPY_VERBOSE):** seedfull window_seeded=17 spec-fail=0 Fill=91% decodeBlock=0.846s (chunks go CLEAN). onlywin seed_hits=**0** (windows UNUSABLE at partition-guess offsets) window_seeded=2 spec-fail=13 decodeBlock=1.06s ≡ prod. onlybnd spec-fail **13→0** (real boundaries kill spec-failures) BUT body still 170MB/s u16, decodeBlock=1.106s ≈ prod ⇒ WALL-NEUTRAL.
- **APPLES-TO-APPLES vs rg (--verbose, both window-absent, SAME 34.5% replaced markers):** rg decodeBlock **0.542s** / Theo-Opt 0.068-0.074s vs gzippy prod **1.067s** / 0.133s ⇒ **rg's u16 marker decode is ~2× FASTER per byte.** rg ties WITHOUT seeding because its marker decode is fast; gzippy ties only by cheating (seedfull = clean, no applyWindow). Even seedfull's CLEAN decode (0.846s) is 1.57× slower than rg's MARKER decode (0.542s).
- **PINPOINTED T8 SUB-LEVER = marker-COMPUTE** (the slow window-absent u16 decode itself, ~2× rg). NOT boundary-alignment (secondary precondition, wall-neutral), NOT spec-failures (wall-neutral). The Phase-0 ISA-L oracle could not see this — ISA-L can't emit u16 markers, so the marker path was never replaced. ⇒ asm/igzip-class inner-kernel work IS in scope HERE, adapted to u16 marker output.
- **INDEPENDENT DISPROOF ADVISOR (synchronous read-only, plans/t8-decompose-advisor-verdict.md):** core verdict UPHELD-WITH-CAVEATS. (Q1) the 2×2 knobs CANNOT separate (a) from (b) — onlywin is DEGENERATE (windows unusable without boundaries by construction ⇒ ≡ prod; its pre-reg self-test FAILED ⇒ void as a windows-only cell, = the COUPLED branch); re-attribute the verdict to **onlybnd + the rg comparison**, not the decomposition. (Q2) onlybnd UPHELD-W-CAVEATS — spec-failures not the cost (clean isolation, wall-neutral). (Q3) the 2× rate gap is FAIR (denominator-matched decodeBlockTotalTime/parallelization, applyWindow separate in both, survives spec-failure removal) — the STRONGEST pillar. (Q4, MOST IMPORTANT) the CEILING is OPTIMISTIC: seedfull removes TWO things — marker decode premium AND the applyWindow serial pass — so it bounds route-(ii) (more clean windows), NOT the faithful route-(i) (fast u16 marker decode, which KEEPS applyWindow like rg). The route-(i) ceiling rests on the **rapidgzip existence proof** (rg: 0.54 decode + ~0.113s applyWindow → 0.13 wall), conditional on gzippy's applyWindow ≈ rg's.
- **BOUNDED CEILING: ≤ T8 1.0× TIE (rapidgzip existence proof), CONDITIONAL on gzippy's apply_window/marker-resolution pass ≈ rg's ~0.113s.** seedfull achieved the TIE but over-removes applyWindow ⇒ optimistic; the conditional bound is the honest one.
- **SCOPED FIX FOR NEXT LOOP (do NOT start — bound-ceiling-first):** an igzip-class u16 marker-decode kernel (asm/inner-kernel techniques adapted to u16 marker output). PLUS the OWED prerequisite measurement before claiming TIE-reachable: time gzippy's apply_window/marker-resolution vs rg's ~0.113s (no existing cell isolates it — needs a fast-marker prototype or a direct apply_window timer). If gzippy's applyWindow ≫ rg's, a marker-COMPUTE-only fix lands SHORT.
- **GUEST STATE:** /root/gzippy src @7bf26096 + oracle overlay + this turn's 2 decompose knobs (seed_windows.rs + chunk_fetcher.rs, applied on guest, NOT committed locally yet). Build /tmp/gzbuild-isal (gzippy-isal, target-cpu=native, byte-exact). Seeds /tmp/seeds.bin (16 windows). Driver /tmp/decompose_measure.sh (use `bash`). No orphan processes.

### SUPERSEDED — PHASE-0 (HEAD 3895a23c +oracle) — T8 BINDER = WINDOW-ABSENT MARKER/SPECULATION PATH, NOT THE ENGINE

### THIS TURN — PHASE-0: dropped a REAL ISA-L engine into the PRODUCTION parallel-SM pipeline and measured the T8 WALL (Measurement PROCESS #3 — engine REPLACEMENT oracle, not isolation-slope extrapolation). This converts the 0.6× engine-PRIMITIVE plateau into an airtight T8 WALL bound.
- **ORACLE (measurement-only, byte-exact, env-gated, NOT production):** `GZIPPY_ISAL_ENGINE_ORACLE=1`
  routes the clean-tail decode in `finish_decode_chunk_impl` through REAL ISA-L FFI
  (`decompress_deflate_from_bit_with_boundaries`, patched igzip), feeding ISA-L bytes/boundaries/
  end-bit through the SAME ChunkData primitives (commit + per-byte CRC + append_block_boundary_at +
  finalize). Pool/consumer/ring/window-publish/scheduling UNCHANGED. ISA-L input bounded to
  `[..stop_hint/8+256KiB]` so each worker decodes only ITS chunk. To run the bulk on ISA-L, windows
  are SEEDED (`GZIPPY_SEED_WINDOWS`, captured at T1) so all 18 chunks are window-PRESENT and reach
  the oracle. PROVEN ISA-L ran: T8 `isal_oracle_chunks=16 isal_oracle_fallbacks=1` (94% real ISA-L).
- **MEASURED (locked guest REDACTED_IP double-ssh, 16c gov=perf turbo-on load 2.7-4.2, measure.sh
  interleaved N=11 CPUS=0,2,4,6,8,10,12,14 RAW=68229982 sha-OK=028bd002…cb410f every run, 2 runs):**
    | contender | T8 wall | vs rg | verdict |
    | rg (rapidgzip 0.16.0)        | 0.134s | 1.000 | — |
    | isal (ISA-L engine, seeded)  | 0.148s | 0.905/0.892 | **TIE** |
    | pure (pure-Rust eng, seeded) | 0.134s | 1.002/0.968 | **TIE** |
    | prod (pure-Rust, NO seed)    | 0.194s | 0.690/0.652 | LOSS |
- **THE LOAD-BEARING RESULT: `pure` (the SLOWER engine) ALREADY TIES rg once windows are seeded;
  `isal` also ties → engine swap is TIE-vs-TIE. The per-thread engine is NOT the T8 wall binder.**
  The whole ~1.5× prod gap collapses to a TIE when the window-absent path is removed. Per-stage
  --verbose: prod decodeBlock SUM 1.048s / Real Decode 0.169s / Fill 77% / body_rate 168 MB/s /
  13 header-speculation failures; pure-seed 0.781s / 0.108s / Fill 90.55% / 0 failures / 0 bootstrap.
  rapidgzip runs the SAME 34.5% replaced-marker workload WITHOUT seeding yet ties (verified rg
  --verbose) → gzippy's window-absent path is the SLOW one, apples-to-apples (NOT a seeding artifact).
- **INDEPENDENT DISPROOF ADVISOR (synchronous read-only, plans/asmport-phase0-advisor-verdict.md):**
  Claim1 (oracle measures igzip-class engine in real pipeline, byte-exact) UPHELD-W-CAVEATS (clean-
  tail only; 1-chunk fallback impurity). Claim2 (engine alone doesn't close T8, TIE-vs-TIE) UPHELD-W-
  CAVEATS (T8-only; engine gap 1.51× is REAL but slack-masked at Fill 90% — NOT at parity). Claim3
  (binder is window-absent path) UPHELD as COARSE localization — sound + not unfair, BUT the seeding
  knob bundles THREE removals: (a) u16 marker decode+resolution, (b) block_finder REAL-boundary
  pre-seed (vs prod partition-GUESS — the project_confirmed_offset_prefetch_gap head-of-line stalls),
  (c) the 13 speculation-failure re-decodes. CANNOT attribute the gain to marker-COMPUTE vs
  boundary-ALIGNMENT vs re-decode. Claim4 (asm port can't move prod wall) directional rec UPHELD at
  T8, strong inference REFUTED: marker-phase decode rate is ON the binding path and was NEVER replaced
  (ISA-L can't emit u16 markers), and T1 (no Fill slack ⇒ engine binds directly) is unaddressed.

### SCOPED TARGET FOR PHASE 1 (the supervisor gate — pick AFTER decomposing the bundle):
- An igzip-class engine ALONE does NOT close the prod T8 wall (pure-Rust already ties seeded). So
  the asm engine port is **NOT the T8 lever** — at T8. It remains plausibly the **T1 lever** (no
  Fill slack, the 1.51× engine gap binds directly) and helps the **marker-phase decode rate** (168
  MB/s, on the binding path, never tested by this oracle). Do NOT abandon it; re-scope it.
- **NEXT PERTURBATION (decompose the Claim-3 bundle BEFORE choosing Phase 1):** seed ONLY the
  block_finder boundaries (no windows) vs seed ONLY windows (prod boundaries). If most of the
  0.69→1.00 delta is boundary-ALIGNMENT, the lever is the block finder / prefetch horizon
  (project_confirmed_offset_prefetch_gap), NOT the asm engine NOR a marker-kernel rewrite. If it's
  marker-COMPUTE, a faster u16 marker kernel (the asm techniques adapted to u16 output) is the lever.
- **HARD WALL BOUND (owed by prior charter, now PAID):** the engine-PRIMITIVE 0.6×-ISA-L plateau
  does NOT bind the T8 WALL — proven by replacing the engine with REAL ISA-L in the production
  pipeline and STILL only tying (engine slack-masked at Fill 90%). The 1.0×-vs-no-FFI FORK is
  NOT forced by the engine at T8 (pure-Rust already ties T8 seeded). The fork may still bite at T1.

### THIS TURN — step (B) executed: built+measured the faithful-u8 engine CEILING vs ISA-L (isolation, bounded)
- **VAR_VI added to benches/engine_isolation.rs** (`decode_var_vi`, x86_64) = VAR_V (faithful-u8
  speculative software-pipelined flat-u8 loop + igzip packed-u32 multi-symbol table, tricks #1/#2/#3)
  PLUS the two REMAINING igzip techniques: (1) **BMI2 BZHI** (`_bzhi_u64`) for the variable-width
  distance extra-bits extraction; (2) **AVX2/SSE MOVDQU wide overlap-copy** back-ref (32B AVX2 bulk,
  16B SSE distance>=16 overlap, RLE memset dist==1). trick #3 (packed-u32 short table) CONFIRMED
  fully exploited (drives the same `LutLitLenCode::decode` packed packets, unpacks up to 3 lit/decode).
- **MEASURED (locked guest REDACTED_IP double-ssh; 16c gov=perf, load ~3.3, turbo on; taskset -c 0;
  N=11 interleaved; native target-cpu ⇒ BMI2+AVX2 LIVE, avx2_detected=true; 2 independent runs, STABLE):**
    | variant | aggregate MB/s | vs ISA-L | per-chunk vs ISA-L |
    | VAR_III ISA-L | 847-851 | 1.000 | — |
    | VAR_V (no BMI2/AVX) | 460-462 | 0.54× | 0.50-0.56 |
    | **VAR_VI (+BMI2+AVX2)** | **504-525** | **0.59-0.62×** | **0.55-0.64** |
  BMI2+AVX2 added ~9-14% over VAR_V but did NOT close the gap. SELFTEST=PASS (iii/i=2.73 ∈ [2.5,3.6]).
- **BYTE-EXACT:** VAR_VI printed an MBps line (never VOID) on EVERY swept chunk ⇒ per the bench gate
  (`exact[k]= o[..n]==scalar && scalar==isal`, engine_isolation.rs:744/802) VAR_VI is byte-identical
  to BOTH the scalar reference AND ISA-L over the full timed window. (Top-line `SHA_ALL_EQUAL=no` is
  the PRE-EXISTING VAR_IV_E234 failures — a separate path NOT touched this turn — not VAR_VI.)
- **PRE-REGISTERED FALSIFIER FIRED → PLATEAU:** VAR_VI ≈ 0.6× ISA-L, ~23pp below the 0.85 PASS line,
  WITH the full igzip stack + inline-ASM intrinsics. ⇒ pure-Rust igzip-class as a STANDALONE ENGINE
  PRIMITIVE is NOT reached on this design.
- **INDEPENDENT DISPROOF ADVISOR (synchronous, read-only, plans/engine-ceiling-advisor-verdict.md):
  PLATEAU UPHELD-WITH-CAVEATS.** Source-verified all 5 techniques LIVE; fast loop (not the careful
  tail) is the timed path; header/table-build symmetric with ISA-L's own header parse; byte-exact
  reasoning airtight. The two minor under-representations (small-overlap 2≤dist<16 copy is scalar not
  igzip-doubling; SHRX compiler-discretionary) + the only asymmetric confound (VAR_VI's final
  `to_vec` ~few-%) ALL cut AGAINST plateau and together lift a "fixed" VAR_VI to at most ~0.65-0.68 —
  STILL ~17-20pp short of 0.85. Structural reason supports plateau: a Rust port routed through a
  `DecodedSymbol` struct-return + `while remaining` unpack carries codegen overhead a hand-scheduled
  asm hot loop does not.
- **LOAD-BEARING ADVISOR CAVEAT (the escalation correction):** the engine-PRIMITIVE ceiling is
  UPHELD, but escalating to "the 1.0× WALL is HARD-BOUNDED at 0.6×" OVERREACHES isolation — that is
  the forbidden extrapolation through an unlocated knee (Measurement PROCESS #3). To hard-bound the
  WALL you must REMOVE the engine stage in the PRODUCTION PARALLEL pipeline and measure, not
  extrapolate the isolation ratio. NOTE: the prior floor-to-floor T8 finding (engine 1.74× at the
  wall, t8-engine-binder-advisor-verdict.md UPHELD) INDEPENDENTLY corroborates the engine gap
  survives to the wall — so the fork is strongly implicated — but the clean WALL hard-bound is still
  owed that one engine-removal perturbation.

### THE FORK (escalate-candidate — supervisor/user call): pure-Rust 1.0× bar vs no-FFI
- **HARD NUMBER (engine primitive, advisor-upheld): pure-Rust+ASM faithful-u8 engine = ~0.6× ISA-L
  in isolation (ceiling ~0.65-0.68 crediting every caveat); the 0.85 igzip-class bar is NOT reached.**
- **The 1.0× WALL-vs-no-FFI fork is REAL.** Two corroborating data points say the engine gap reaches
  the wall: (i) the floor-to-floor T8 1.74× engine gap (advisor-upheld this campaign); (ii) the
  constant ~1.70× gzippy↔rapidgzip ratio at BOTH T1 and T8 (per-thread-throughput signature).
- **What is NOT yet a clean hard-bound:** the WALL number under an ENGINE-REMOVAL oracle in the
  production parallel pipeline (Rule 3). Recommended BEFORE a final fork decision: run that
  perturbation (replace the per-thread decode with a no-op/ISA-L oracle, measure the T8 wall) — if
  the wall stays ~1.7× off rapidgzip the fork is hard-forced (no-FFI cannot reach 1.0×); if it ties,
  a shared serial stage gates and pure-Rust CAN still tie despite the 0.6× engine.

### NEXT (decision point — supervisor gate):
- Either (a) ESCALATE the fork now with the engine-primitive hard number (~0.6× ISA-L, PLATEAU) +
  the corroborating wall evidence, accepting the advisor's caveat that the wall hard-bound is owed
  one more perturbation; OR (b) FIRST run the production-pipeline engine-removal oracle to convert
  the engine ceiling into a clean WALL hard-bound, then escalate. The owner recommends (b) is cheap
  and removes the last ambiguity, but the engine-primitive PLATEAU itself is settled.

---
## SUPERSEDED — PRIOR CURRENT STATE (2026-06-07, HEAD f8260aa8) — T8 BINDER RE-LOCATED to the ENGINE; the "serial/consumer-wait" binder is REFUTED (floor-to-floor + advisor-UPHELD)

### THIS TURN — step (A) ceiling-bound; the prior "binder = serial/consumer-wait" was a UNIT ERROR; binder is the per-thread DECODE ENGINE
- **THE PRIOR CHARTER BINDER ("decode floor 0.118s ALREADY ≈ rapidgzip's wall 0.130s, so the
  whole 1.7× gap is scheduling/consumer-wait") IS REFUTED — it was a UNIT ERROR (advisor UPHELD,
  plans/t8-engine-binder-advisor-verdict.md).** It compared gzippy's decode FLOOR (0.118s) to
  rapidgzip's WALL (0.130s). The correct comparison is FLOOR-TO-FLOOR: rapidgzip's own
  Theoretical-Optimal is 0.068s, NOT 0.130s. gzippy 0.118 vs rapidgzip 0.068 = **1.74× engine gap.**
- **FIRST-HAND apples-to-apples --verbose pool stats this turn (locked guest REDACTED_IP double-ssh,
  16c gov=perf, box load ~2.5 ⇒ INTERNAL SPANS not wall absolutes; gzippy-mk2 byte-exact
  028bd002…cb410f path=ParallelSM; rapidgzip 0.16.x --verbose; 3 runs each, STABLE), T8 silesia:**
    | metric | gzippy | rapidgzip | ratio |
    | decodeBlock (SUM/workers) | 0.93s | 0.50s | **1.86×** |
    | Theoretical-Optimal (÷8) | 0.118s | 0.068s | 1.74× |
    | Total Real Decode Duration | 0.139s | 0.086s | 1.61× |
    | std::future::get (consumer wait) | 0.077-0.082s | 0.062-0.067s | ~1.25× |
    | Pool Fill Factor | 85% | 78% | — |
- **BINDER = the per-thread DECODE ENGINE** (decodeBlock 1.86×; body_rate 269 MB/s vs rapidgzip's
  ~424 MB/s ISA-L = 1.58× raw + speculative/marker overhead, BOTH engine). The consumer future::get
  gap (1.25×) is a MINORITY and largely DOWNSTREAM (the consumer waits longer because each chunk
  decodes slower). This matches the long-observed CONSTANT ~1.7× ratio at BOTH T1 and T8 (flat-
  across-T = the signature of a per-thread throughput gap, which the charter itself noted).
- **CEILING-BOUND METHOD NOTE (Rule 3): the decode-bypass + sleep-decode oracles are CONFOUNDED**
  (decode-FREE wall was 3.6-5.5× SLOWER than real decode — they bypass the buffer pool, do fresh
  full-size zeroed allocs/faults per chunk, hold ≤33 ChunkData/660MB live, single-thread CRC 212MB
  un-overlapped). The valid ceiling instrument is the FLOOR-TO-FLOOR --verbose span comparison.
- **VENDOR SOURCE-VERIFIED (BlockFetcher.hpp:246-329, this turn):** rapidgzip's get() ALSO pumps
  prefetchNewBlocks() in a `while(wait_for(1ms))` loop during the future wait (:314-316), exactly
  as gzippy (chunk_fetcher.rs:1289 Lever H). The consumer-overlap STRUCTURE is already faithfully
  ported; future::get is non-zero in BOTH. There is NO missing overlap mechanism to port.
- **PROD DECODE-MODE SPLIT (T8): finished_no_flip=16, window_seeded=2, flip_to_clean=0.** 16/18
  chunks take Engine M's speculative marker-bootstrap-then-u8-direct-tail path (window-absent at
  high T, faithful — rapidgzip is also ~window-absent at runtime). The engine front IS the bulk path.

### NEXT (per PROCESS — bottleneck is the ENGINE; step (B) = build+measure the faithful-u8 ceiling):
- **The advisor's load-bearing caveat (D-D): the ENGINE-BENCH-ROUND-2 "2.4× plateau" that earlier
  declared pure-Rust+ASM unreachable was measured on the DISCREDITED u16-RING architecture. It does
  NOT bound the CURRENT faithful u8-direct flip-in-place engine (landed fc1c965b). So the
  pure-Rust→1.0× question is OPEN, NOT settled by that plateau.**
- **USER-CONSTRAINT FORK IMPLICATED (advisor-flagged, escalate-candidate):** is the 1.0× bar
  reachable in pure-Rust+ASM (no FFI) given the engine is 1.86× ISA-L? Must be resolved by BUILDING
  + measuring the faithful-u8 engine ceiling vs ISA-L on the production speculative path, NOT by
  extrapolating the invalid u16 plateau. Lever: igzip-class inner-Huffman (packed-u32 short table,
  speculative 8-byte literal store + next-sym/next-dist preload pipeline, BMI2 SHLX/SHRX/BZHI,
  MOVDQU overlap-doubling copy, slop-margin headroom) on the u8-direct ring — authorized in scope.
- Kernel is now the confirmed binder at BOTH T1 (was always) AND T8 (this turn). The consumer-wait
  direction is DESCOPED (minority + already-faithful structure).

---
## SUPERSEDED — PRIOR CURRENT STATE (2026-06-07, HEAD fb3baec0) — "T8 BINDER = SERIAL/CONSUMER-WAIT" (REFUTED above as a unit error)

### THIS TURN — step (A) executed; u16-path premise FALSIFIED; binder LOCATED to the serial/consumer term
- **u16-path "biggest prize" premise is FALSIFIED at the source (advisor UPHELD,
  plans/u16-ceiling-advisor-verdict.md).** The "58.6% u16" came from the MIS-NAMED counter
  `BOOTSTRAP_POST_FLIP_U16_BYTES` (gzip_chunk.rs:97/:1302): it increments when
  `!block.contains_marker_bytes()` — i.e. it counts bytes in marker-FREE blocks, which since
  fc1c965b decode u8-DIRECT (marker_inflate.rs:1397-1401 ring_modulus=U8_RING_SIZE, :1685
  ring8.write). The counter NAME + doc (gzip_chunk.rs:91-96) are STALE/inverted. The genuine
  u16-`<true>` fraction is the INVERSE ≈ 42.5% (the pre-flip prefix each speculative chunk must
  decode before 32KiB clean accumulates → flip). NOT "the bulk of bytes on a slow path."
- **CAUSAL PERTURBATION (this turn; new GZIPPY_SLOW_MARKER_MODE u16-path knob, commit fb3baec0;
  byte-exact OFF/marker100/clean100/marker100+sleep all = 028bd002…cb410f; locked guest
  REDACTED_IP double-ssh, 16c gov=perf, measure.sh interleaved sha-OK, RAW=211968000, T8
  CPUS=0,2,4,6,8,10,12,14, N=11; box load 3-5, interleaved-relative is load/turbo-robust):**
    - CLEAN +100% spin → +27%; CLEAN +100% SLEEP control → +27% (IDENTICAL ⇒ NOT a turbo
      artifact); CLEAN +200% SLEEP → +55%. ⇒ clean u8 decode-compute GENUINELY gates ~27% of T8
      wall (freq-neutral, ~linear). (Supersedes the prior "~18-22%" — that was on a different
      box/run; the freq-neutral confirm makes ~27% the number.)
    - MARKER +200% spin → +21%; MARKER +200% SLEEP control → +7% (does NOT survive the control).
      ⇒ u16-marker decode-compute is a MINORITY: ~3.5-14% of T8 wall (advisor range; point est
      ~3.5%, biased low by calibration D1 + event-coverage D3, high-single-digits most likely).
    - T1 MARKER +100% → +0% / +200% → +4% (near-flat: at T1 ~all chunks window-seeded clean, u16
      barely runs; the knob fires ∝ u16 bytes ⇒ near-flat validates it).
- **BINDER LOCATED (not residual) from the GZIPPY_VERBOSE pool trace, first-hand this turn:**
  decodeBlock(all workers)=0.936s → **Theoretical-Optimal (÷8) = 0.117s**; Total Real Decode
  Duration (pool phase span) = 0.147s (Fill 79%); **std::future::get (in-order consumer wait) =
  0.077s**; header_ms=24.0 (~2.6% of decode — the D4 header/table-build caveat is quantitatively
  TINY); full wall this run 0.183s (interleaved best ~0.183-0.221s). 3-way anchor: gzippy-mk ≈
  varv (1.018× TIE), rapidgzip 0.130s = 1.70×.
- **DECISIVE DECOMPOSITION:** gzippy's perfectly-parallel decode floor (Theoretical-Optimal
  0.117s) is ALREADY ≈ rapidgzip's ENTIRE wall (0.130s). The whole 1.70× gap is the
  scheduling/serial term: pool fill gap (0.147-0.117 ≈ 0.030s) + in-order consumer `future::get`
  head-of-line wait (~0.077s) ≈ **~0.10s of serial/overlap = the dominant T8 binder.** rapidgzip
  ties DESPITE the same engine gap by OVERLAPPING decode under scheduling (memory
  project_confirmed_offset_prefetch_gap: gzippy consumer cold-stalls in-order get; rapidgzip
  joins in-flight). **Conclusion #2 advisor caveat: the residual is scheduling/serial + a small
  header/bandwidth term, now MEASURED (header ~2.6%), not eliminated-by-residual.**

### NEXT (per PROCESS — bottleneck is the serial/consumer-wait term, ~0.10s):
- **FIX the in-order consumer `future::get` head-of-line wait (~0.077s) + pool fill gap (~0.030s).**
  This is charter binder #2 and the `project_confirmed_offset_prefetch_gap` memory: make gzippy's
  consumer JOIN an in-flight decode instead of cold-stalling (rapidgzip GzipChunkFetcher.hpp
  consumer loop :1419-1469 cold-get + :1535-1740 serial window-publish chain). CAUTION: the
  prior `placement-port` GATE FAILED (offset-supply was a non-divergence); the OPEN distinct
  question per that gate is the PREFETCH-HORIZON / dispatch-depth (decode_NOT_STARTED stalls =
  guess-prefetch never dispatched DEEP ENOUGH AHEAD), NOT offset supply. Bound the ceiling first
  (Rule 3): an ORACLE that removes the consumer wait (e.g. unbounded look-ahead / pre-resolved
  futures) → measure the whole-system wall; if it lands ~0.12s it confirms the tie is here.
- Keep the inner kernel as the confirmed T1 lever AND a real ~27% T8 contributor (freq-neutral),
  but it is NOT the T8 path to 1.0× — closing the serial term gets to ~rapidgzip's wall alone.
- u16-prefix ceiling: KEEP a prefix-removal oracle in reserve (Rule 3, advisor D6) before fully
  abandoning — a faster prefix COULD let chunks flip sooner / consumer catch up. But it is a
  minority term; do NOT lead with it.

---
## PRIOR STATE (2026-06-07, HEAD 9b674651) — T8 BINDER RE-IDENTIFIED (causal + advisor-upheld) [SUPERSEDED above re: u16]
- gzippy-native is FAITHFUL u8 (u8-direct flip-in-place clean tail landed byte-exact). VAR_V
  speculative pipeline committed (byte-exact TIE, kept per 7a).
- Whole-system wall (locked guest trainer=REDACTED_IP via -J neurotic, 16c gov=performance
  no_turbo=1, measure.sh interleaved sha-verified, RAW=211968000): T8 gzippy ~0.226s vs rapidgzip
  ~0.137s = **1.655× gap** (varv vs base TIE, sha 028bd002…cb410f OK). Reproduced this turn.

- **CHARTER CORRECTION (this turn, causal + disproof-advisor UPHELD-WITH-CAVEATS,
  plans/t8-binder-advisor-verdict.md): the prior "constant 1.70× = pure per-thread decode gap,
  inner Huffman kernel is the ONLY lever to 1.0× at T8" is REFUTED AT T8.** Established via the
  slow_knob causal perturbation (byte-transparent, frequency-neutral sleep control confirms not a
  turbo artifact; site fires ∝ clean bytes):
    - T1 (CPUS=0): spin100 (doubles per-thread decode-compute) → +83% wall (off 0.533→0.974s).
      => decode-compute GATES ~83% of the T1 wall. **Kernel is the confirmed T1 lever.**
    - T8 (8 pinned cores): spin100 → +14–22% wall; spin200 → +45%; sleep100 control +20% (≥ spin).
      => per-thread CLEAN decode-compute gates only ~18–22% of the T8 wall directly.
    - COVERAGE CONFOUND reconciled first-hand: slow_knob is CLEAN-mode only (const-folds to 0 on
      the marker <true> path). Clean-loop hits T1=38.7M vs T8=28.4M (T8 = 73% coverage) ⇒ ~27% of
      T8 decode events run in MARKER (u16) mode, uncovered. Coverage-corrected decode-compute
      ceiling at T8 ≈ ~25–30% of wall (advisor: plausibly up to ~45% with Rule-3 unbind slack).
      EITHER WAY decode-compute is a MINORITY of the T8 wall; ≥~55–75% is OTHER.
- **THE T8 BINDER (the OTHER ≥55–75%), from the GZIPPY_VERBOSE trace (first-hand this turn):**
    1. **u16 post-flip / marker path = 58.6% of decoded BODY bytes** (`post_flip_u16_bytes
       =118.6M, "Design-B1 prize"`). The bulk of production bytes at T≥2 flow through the slower
       u16 marker→drain path, NOT the clean u8 fast path the kernel/VAR_V optimized. This is the
       largest single named term and is why VAR_V's clean gain was absorbed + the slow_knob barely
       moved T8. body_rate blended 286 MB/s; Speculation failures header=14/19.
    2. **Pool scheduling + serial tail:** Theoretical-Optimal 0.127s → Real-Decode-Duration 0.162s
       (~28% pool inefficiency, fill 73–83%, Prefetch dispatch saturated ~51/60) → wall ~0.22s
       (another ~0.06s SERIAL outside the pool: in-order consumer publication / drain / CRC).
       Corroborates memory project_confirmed_offset_prefetch_gap (head-of-line stalls ~40% T8).
- Window-discard lever: FALSIFIED (prior turn; window seeded when available; T≥2 window-absent is
  faithful rapidgzip behavior).
- **NEXT (re-pointed per the PROCESS — bottleneck moved off the clean kernel at T8):** the two T8
  binders above. Recommended order: (A) bound the u16-path ceiling with an ORACLE removal (NOT the
  slope) — if the 58.6% u16 body decoded at clean-path rate, how much wall drops? rapidgzip ties
  its wall DESPITE the same engine gap via in-flight overlap, so this is likely the bigger prize;
  (B) the pool-scheduling/serial-tail SERIAL-WORK vs DECODE-WAIT decomposition. Keep the inner
  kernel as the confirmed T1 lever (not abandoned), but it is NOT the T8 path to 1.0×.
- NO new build this turn (perception + causal ID + advisor only). Tree clean, no orphans.

---
## USER DECISION 2026-06-07 (fork resolved): TRANSLITERATE igzip's FULL AVX2 ASM KERNEL
The pure-Rust engine ceiling is bounded (advisor-upheld): faithful-u8 + the FULL igzip technique
stack + inline-ASM intrinsics (BMI2 BZHI, AVX2/SSE overlap copy, packed-u32 table, speculative
pipeline) = VAR_VI ~0.60× ISA-L in isolation (~515 vs ~849 MB/s) — high-level techniques do NOT
reach hand-tuned igzip asm. User chose: pursue **pure-Rust no-FFI 1.0× by transliterating igzip's
ACTUAL assembly instruction-for-instruction** (our own inline Rust asm — NOT C-FFI). Honors 1.0× +
no-FFI + faithfulness if it lands. This is a MULTI-SESSION project; own it in byte-exact phases.

ASM-PORT PROJECT PLAN (the owner owns; phased, prove-before-the-big-build):
- **PHASE 0 (scope the target — do FIRST, cheap):** an ISA-L-in-pipeline WALL oracle — drop an
  igzip-class engine (real ISA-L FFI, MEASUREMENT-only) into gzippy's PRODUCTION pipeline and
  measure the T8 WALL vs rapidgzip. Tells us: does an igzip-class engine ALONE tie in gzippy's
  real pipeline (⇒ the asm port is sufficient, target = match igzip rate), or do production
  overheads (ring/wrap/resumable/CRC — which absorbed VAR_V) ALSO cap it (⇒ the port must
  integrate into a FLATTENED clean path)? This converts the 0.60× engine-primitive plateau into an
  airtight WALL bound (PROCESS #3) AND scopes the transliteration so it can't be absorbed like VAR_V.
- **PHASE 1+ (the transliteration):** port igzip_decode_block_stateless_{01,04}.asm → inline Rust
  asm, integrated per Phase-0's finding (flatten the path if needed), in byte-exact + wall-measured
  phases, each advisor-gated. Target: production T8 wall ties rapidgzip (~0.13s same-host).

---
## PHASE-0 RESULT 2026-06-07 (advisor-upheld) — ASM PORT IS NOT THE T8 LEVER; RE-SCOPE
ISA-L-in-pipeline wall oracle (commit 5e9905c8, proven ISA-L ran 16/18 chunks, byte-exact):
T8 seeded — pure-Rust 0.97-1.00× TIE, ISA-L 0.90× TIE (TIE-vs-TIE); unseeded production
0.65-0.69× LOSS. ⇒ The pure-Rust ENGINE ALREADY TIES T8 once windows are seeded; the engine
plateau (0.6× isolation) does NOT bind the T8 wall. **The T8 lever is the WINDOW-ABSENT
marker/speculation path** (16/18 chunks window-absent @168 MB/s + 13 header-speculation
failures; rapidgzip runs the same ~34.5% marker workload without the penalty). The T8 1.0× tie
is reachable in PURE-RUST, no FFI, WITHOUT the asm transliteration.
RE-SCOPE (no constraint fork — same goal, pure-Rust): the asm engine port is the **T1 lever**
(T1 has no scheduling slack ⇒ engine binds directly, ~83% of T1 wall) — keep for T1, defer for
T8. The **T8 lever is the window/marker/boundary path**. Advisor caveat: "seeding" BUNDLES three
removals — (1) u16 marker-compute, (2) block-finder real-boundary vs partition-guess ALIGNMENT,
(3) 13 speculation-failure re-decodes. NEXT: DECOMPOSE the bundle (seed-only-boundaries vs
seed-only-windows) to pinpoint the precise T8 sub-lever before fixing — likely boundary-ALIGNMENT
= block-finder / prefetch-horizon (project_confirmed_offset_prefetch_gap), faithfully ported.

---
## PROCESS ADDENDUM — IT'S A RATCHET, not a flip-flop (user-set 2026-06-07)
As performance improves, the binder WILL oscillate between the per-thread engine and the
sequential/scheduling terms — that is the EXPECTED shape of whole-system bottleneck-following,
not thrashing. Fix the current binder, it migrates to the next-largest term; fix that. The only
metric of progress is the WHOLE-SYSTEM WALL RATCHETING DOWN. A banked byte-exact wall-moving
change (e.g. merge-removal +12%, wall 0.65×→0.73× rg) is a ratchet tooth — it does NOT come back,
even though the binder then moves on. Do NOT treat binder migration as a problem to avoid; keep
banking ratchet teeth.
DISTINCTION (the real discipline): separate (a) genuine binder MIGRATION (healthy ratchet) from
(b) FALSE flips caused by measurement error (the unit error; the misnamed counter read backwards
twice). (a) is fine and expected. (b) is the enemy — kill it: every binder claim must come from a
CAUSAL perturbation on the WALL (not producer-side attribution), and counters must be named for
what they actually count. Make every flip a REAL migration.

---
## PROCESS FIXES (supervisor coach review, plans/SUPERVISOR-FEEDBACK.md, 2026-06-07) — BINDING
1. **The decider ORACLE gates each loop.** Before ANY binder-MECHANISM claim or fix, run the
   registered removal oracle for that binder (for the current scheduling direction: the
   `GZIPPY_PERFECT_OVERLAP` ceiling — NEVER YET RUN; it is the registered decider and the current
   strategy rests on it unmeasured = a live Rule-3 violation). No mechanism claim without the
   oracle.
2. **Causal-perturbation-first; attribution is a FOOTNOTE.** Every binder claim must LEAD with a
   causal perturbation on the WALL. SUMs / ratios / Fill% / counters are hypothesis-only and have
   repeatedly produced inverted binders (the misnamed counter read backwards twice). Never let
   attribution be the verdict.
3. **No STOP/TIE/"done" without a validated removal oracle.** Two prior "victories" were reversed
   (2026-06-02 STOP-EARNED; 2026-05-29 rescind on a broken oracle). Don't repeat.
ALSO (coach-corrected facts): engine is CONCLUSIVELY NOT the T8 binder (seeded-pure TIE 1.002× +
ISA-L oracle TIE-vs-TIE) — do NOT re-test the engine at T8 (that is the ~40% wasted re-derivation
in the engine↔scheduling oscillation); the T8 lever is the named head-of-line stall
([[project_confirmed_offset_prefetch_gap]], "fixable not architectural"). 0.73× is NOT "best ever"
(June-2 ~0.85× was a different ISA-L-product basis) — never frame it as a regression OR a record.
