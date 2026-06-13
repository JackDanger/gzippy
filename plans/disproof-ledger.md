# DISPROOF LEDGER — gzippy decode campaign (WARM RED-TEAM, role 2)

Standing, COMMIT-PINNED ledger of inference judgements. Gates the INFERENCE
("does a bankable number SUPPORT the claim/lever?"), NOT measurement hygiene
(Steward owns that). Per CLAUDE.md rule 1: a lever is confirmed ONLY by a causal
perturbation (slow-injection slope + frequency-neutral control, or a removal
oracle); ATTRIBUTION is never a verdict => attribution-only = UNCONFIRMED.

Seeded 2026-06-08 against **HEAD d56cb0f5** (branch reimplement-isa-l). Every
entry is pinned to the commit/file it rests on; if that code moved, the entry
reverts to OPEN until re-confirmed. Source-verified first-hand where a file:line
is cited.

STALENESS RULE (advisor fix #2): a DISPROVEN entry is binding only while the cited
region is unchanged. A re-measured TIE is binding only while the binary in that
region is unchanged (a dist-cache shrink / ShortBitsCached port can revive a dead
direction — do NOT freeze a valid lever behind a stale disproof).

---

> **DISPROOF LEDGER USAGE CONVENTION — annotations added 2026-06-13**
>
> **Every entry holds ONLY within its measured CONTEXT** — commit/bin-sha, corpus, arch,
> thread count, and code state at time of measurement — and at the CONFIDENCE its evidence
> supports. A "DISPROVEN" verdict is a time-stamped finding, NOT an eternal law. Cite an
> entry ONLY with its scope tag. RE-MEASURE before relying on any entry whose cited code
> region has changed since measurement.
>
> **Evidence tiers** (for CONFIDENCE annotations):
> - **removal-oracle**: strongest — cited region removed; wall-neutral ⇒ slack confirmed
> - **causal perturbation + freq-neutral control**: strong — monotonic slope survives sleep swap
> - **causal perturbation (spin only)**: moderate — may be turbo artifact; freq-neutral control owed
> - **attribution / model fit / perf-record shares / S+W/T regression**: weak — directional only, NOT a verdict; CLAUDE.md rule 1 forbids attribution-alone conclusions
> - **Δ < spread**: TIE verdict on that run only, NOT a refutation of the direction
>
> **STATUS tags** used in annotations below:
> - **HOLDS-AT-HEAD**: re-confirmed or structurally immutable at current HEAD
> - **SCOPED**: holds in its stated context; not re-verified at HEAD — valid to cite WITH scope
> - **SUPERSEDED**: a later measurement overturned it; annotation names the superseding evidence
> - **STALE-RISK**: cited code region IS modified since measurement; verify before citing as current
>
> **Numbering notes**: two entries share the label DIS-18 (first at the per-symbol-attribution
> section ~line 488; second at the T4-curve/Amdahl section ~line 729) — they are distinct findings;
> always cite by content, not number alone. DIS-29 appears before DIS-28 in file order (each was
> added chronologically; the higher number was written first). Neither anomaly represents an error
> to fix here; be aware when searching by number.

---

## A. BINDING BAR / GOAL SHIFTS

| id | entry | status | pin |
|----|-------|--------|-----|
| BAR-1 | **TIE = >=0.99x at EVERY thread count** (T1, T4, T8, T16…), interleaved + sha-verified, quiet box. 0.88x is NOT a tie; "within spread" / "ties at T8" is REJECTED. | **BINDING (user-set 2026-06-08)** | memory project_tie_bar_99pct_all_threadcounts.md |
| BAR-0 | SUPERSEDED prior framings: "TIE within spread", "ties at T8 (1.030x) => done". These let an 0.88x cell pass and are VOID under BAR-1. The GOAL #2 "TIES rg at T8" completion line (orchestrator-status:165) is a PASS on T8 ONLY — NOT a build-level done under BAR-1. | SUPERSEDED by BAR-1 | orchestrator-status.md:165-179 |
| BAR-2 | Scorecard under BAR-1 (RE-ESTABLISHED first-hand 2026-06-08, frozen guest, N=11, env-unset PRODUCTION, sha=OK, path=ParallelSM, vs rg 0.16.0): **gzippy-ISAL** T1 0.899x / T4 0.900x / T8 0.990x(TIE, 9% spread) — isal_chunks 16/14/14, fallbacks 1/0/0. **gzippy-NATIVE** T1 0.608x / T4 0.761x / T8 0.915x — isal_chunks=0. isal is 14-48% faster than native at every T (ISA-L active, NOT dormant). **Neither build is done under BAR-1** (isal passes T8 only at the 0.990 threshold; both lose T1/T4). Low-T is the headline. | CURRENT (verified d56cb0f5) | plans/isal-dormancy-advisor-verdict.md; bins 2d317027/a42d4600 |
| BAR-3 | Goal reinstated 2026-05-30: faithful STRUCTURAL convergence to rapidgzip + pure-Rust SOLE decode path, C-FFI off the decode graph. The 2026-05-29 rescind (rested on the broken clean-window oracle) is REVERSED. asm-engine is the user's gated call; both builds lose T4. | CURRENT | CLAUDE.md Goal; INSTR-1 below |

**SCOPE/STATUS ANNOTATIONS — Section A (2026-06-13)**

| id | SCOPE | LEDGER-STATUS |
|----|-------|---------------|
| BAR-1 | User-set 2026-06-08; CLAUDE.md + memory/project_tie_bar_99pct_all_threadcounts.md | **HOLDS-AT-HEAD** — user-set binding rule; unchanged |
| BAR-0 | Already labeled SUPERSEDED by BAR-1 in the entry above | **SUPERSEDED** (by BAR-1; no new evidence needed) |
| BAR-2 | 2026-06-08, d56cb0f5, neurotic guest 10.30.0.199, silesia, T1/T4/T8, bins 2d317027/a42d4600 | **SCOPED** — scorecard verified at d56cb0f5 on neurotic; not re-verified on solvency at HEAD. The "CURRENT (verified d56cb0f5)" label is STALE as of this annotation — the entry's numbers are valid in their context but should not be cited as the current scorecard without re-measurement on solvency. CONFIDENCE: N=11 interleaved, sha=OK, production env-unset — strong methodology for its measurement date |
| BAR-3 | User-set 2026-05-30; CLAUDE.md Goal | **HOLDS-AT-HEAD** — user-set goal; unchanged |

---

## B. DISPROVEN CLAIMS (+ mechanism) — pinned; do NOT re-attempt as-stated

| id | disproven claim | MECHANISM of disproof | pin |
|----|-----------------|------------------------|-----|
| DIS-1 | "Inline-asm transliteration of igzip captures the engine share." | **NO-GO via per-symbol asm<->Rust re-entry spill.** VAR_VII inline-asm hot loop measured 78 MB/s (0.276x ISA-L), ~0.75x of NAIVE scalar; rate FALLS as asm coverage rises => the per-symbol asm↔Rust re-entry (LLVM barrier ×300-460K/chunk, 4 regs spilled to `bits`) dominates. Byte-exact but slower. Advisor disproof ×2, pass-2 signed off. | benches/engine_isolation.rs VAR_VII; commit 690941f3; plans/phase2-inline-asm-advisor-verdict{,-pass2}.md |
| DIS-2 | "The clean-window oracle shows gzippy is already at rapidgzip parity (isal 2035 ≈ rg 2067)." | **Broken instrument: it silently RE-RAN the full bootstrap.** Premise dead; the decompose-a-slice-and-shave loop it licensed is void. Fixed 64eb6df. | CLAUDE.md Goal §; commit 64eb6df |
| DIS-3 | "isal_oracle_chunks= coverage assert proves engine-isolation ran ISA-L." | **Grep-bug: binary emits `isal_chunks=N isal_fallbacks=M`, NOT `isal_oracle_chunks=`** (chunk_fetcher.rs:870-874) => the assert hard-failed coverage-unreadable on EVERY prior attempt — a stale self-test reading as passing. Script-only fix landed; binary untouched. | chunk_fetcher.rs:870-874; orchestrator-status JOB 1 :52-55 |
| DIS-4 | "ocl_cf is 0.945x (≈ ties rg) at T8" (the PESSIMISTIC banked number) AND "ocl_cf is 0.997x". | Both REFUTED. Hardening caught the box THAWED (no_turbo=0, watchdog expired) — the source of the 0.945↔0.989 drift. Banked value re-set to "≈TIE / >=0.945x" at T8; do NOT bank 0.997x (one-run, load1.63). | commits e4389f05/f216c691; plans/oracle-hardening-advisor-verdict.md |
| DIS-5 | "Output-overlap writer is the faithful T8 binder lever (0.88x ceiling)." | **NON-FAITHFUL + sub-parity.** There are TWO T8 binders not one; the 0.88x ceiling is a CONSTRUCTION (built by DESTROYING overlap), not measured. /dev/null comparator showed it. Output-overlap REFUTED as the path. | orchestrator-status :351-392; plans/output-reconciliation-advisor-verdict.md |
| DIS-6 | "Offset-supply / re-target-the-overshot-index closes the confirmed-offset prefetch gap." | **Premise FACTUALLY WRONG — gzippy ALREADY re-targets** (gzip_block_finder.rs:180-182, chunk_fetcher.rs:1306/:1431; needs_confirmed_offset zero hits). 3 prior gzippy attempts FAILED; consumer-confirmation prefetch is DEAD by measurement (~1-chunk lead too short; decode IN-FLIGHT-NOT-DONE when consumer arrives). Do NOT re-attempt offset-supply. | memory project_confirmed_offset_prefetch_gap §FAILED/CONCLUSIVE; commit 43f1685/e52b0fc2 |
| DIS-7 | "Free-decode oracle (Oracle-C) bounds the engine ceiling." | **DEGENERATE: free decode also frees windows => publish-chain collapses** => its 0.4-0.7s was GREY/uninterpretable. Resolved by the clean-only oracle (publish-chain PRESERVED). | memory project_pregate_placement; commit b8a38e64 |
| DIS-8 | "combine_crc costs 62ms serial CRC." | Phantom: a nested-span DOUBLE-COUNT; it is an O(1) combine of worker-computed CRCs. (Canonical example of hand-script attribution manufacturing a phantom — CLAUDE.md rule 8.) | CLAUDE.md rule 8 |
| DIS-9 | "C2 21ms non-engine residual is a faithful low-risk scheduling tooth." | **All C2 sub-terms FLAT-or-small** under 3 removal oracles (DEPTH dead by removal; TARGETING 1-chunk). NOT a faithful low-risk scheduling lever. Advisor signed off ×2. | orchestrator-status :243-264; commit f98af1f; plans/c2-residual-disproof-verdict.md |
| DIS-10 | "The cited rapidgzip isal.hpp:392-405 read_in resync is the fix for the stored/fixed clean-tail coverage gap." | **DOUBLY WRONG**: (a) isal.hpp:392-405 is readBytes() — a byte-aligned FOOTER reader, NOT a block resync (real mechanism = readStream(), isal.hpp:255-360); (b) the FFI wrapper ALREADY records boundaries (40,960 on the repro) => the production "ZERO boundaries on stored/fixed" comment is EMPIRICALLY FALSE for SYNC_FLUSH. Real fix = relax gzip_chunk until_exact EXACT-match accept (a NEW gated turn). | orchestrator-status JOB 2 :3-44; worktree commits 8c87cc24+7695463d |
| DIS-11 | "GZIPPY_PERFECT_OVERLAP oracle refutes F1 (the T8 overlap-TIE is unreachable)." | Oracle built BACKWARDS — an upper bound built by DESTROYING overlap CANNOT falsify the TIE-is-reachable claim. F1 UNDECIDED by that run. | orchestrator-status :827-852; plans/perfect-overlap-advisor-verdict.md |
| DIS-12 | "gzippy-isal's clean tail is pure-Rust (StreamingInflateWrapper/resumable.rs); build.rs:98-110 is correct that BOTH topologies decode the clean tail in pure Rust and that REAL ISA-L FFI is reachable ONLY under GZIPPY_ISAL_ENGINE_ORACLE=1." | **REFUTED — source-verified first-hand at HEAD d56cb0f5; the build.rs comment was the WRONG one and is now FIXED.** Chain: Cargo.toml:84 (gzippy-isal ⊇ pure-rust-inflate+isal-compression) → build.rs:110 `isal_clean_tail` set → FlipToClean (gzip_chunk.rs:1169) → `finish_decode_chunk_with_inexact_offset` (:1185, `allow_isal=true` :630) → gate `allow_isal && isal_engine_oracle_enabled()` (:669) → `isal_engine_oracle_enabled()` defaults env-unset to `cfg!(isal_clean_tail)`==true (:154-161) → REAL ISA-L FFI `decompress_deflate_from_bit_into` (`finish_decode_chunk_isal_oracle` :205-). So on the gzippy-isal build the PRODUCTION clean tail IS ISA-L FFI; `StreamingInflateWrapper` (resumable.rs) is the COUNTED fallback only (ISAL_ENGINE_ORACLE_FALLBACKS, asserted ==0 on the all-dynamic parity corpus). The banked GOAL #2 (19add96c) is the source-correct description; the env var is an OVERRIDE not the only enable. RECONCILIATION: build.rs:98-110 comment corrected (byte-transparent) on branch fix/buildrs-isal-comment; the "pure-Rust isal tail" claim was a stale-comment error (it over-generalized the gzippy-native truth). **NOW EMPIRICALLY CONFIRMED (DIS-13): env-unset isal binary measures isal_chunks=14/14 fallbacks=0.** | build.rs:98-110 (fixed); gzip_chunk.rs:154,161,205,630,669,1169,1185; Cargo.toml:84; plans/converge-bootstrap-advisor-verdict.md claims 1-2; plans/isal-dormancy-advisor-verdict.md |
| DIS-13 | "ISA-L is RUNTIME-DORMANT in gzippy-isal production at HEAD (isal_chunks=0; isal T4 654ms=0.757x==native; the asm target is the marker bootstrap for BOTH builds; GOAL #2 overstated)." (the FRESH/UNBANKED residual-attribution claim) | **REFUTED by first-hand build+measure — the fresh number MEASURED A gzippy-NATIVE BINARY mislabeled as gzippy-isal (mislabeled-binary class, CLAUDE.md rule 4).** At HEAD d56cb0f5, env-unset (oracle UNSET, parity.sh-scrubbed), frozen, N=11, sha=OK, path=ParallelSM: gzippy-ISAL (bin 2d317027) isal_chunks=14/14 fallbacks=0 @T4/T8, 16/1 @T1; wall T1 0.899x / T4 0.900x / T8 0.990x(TIE). gzippy-NATIVE (bin a42d4600) isal_chunks=0; wall T1 0.608x / T4 0.761x / T8 0.915x. The fresh "isal" counters (isal_chunks=0 flip_to_clean=12 finished_no_flip=4 window_seeded=2 clean_flipped 2%) and wall (654ms=0.757x) are BYTE-FOR-BYTE the gzippy-NATIVE signature I reproduced (native T4 = 652ms=0.761x, identical counters). isal_chunks=14 is structurally IMPOSSIBLE on native (the stub `finish_decode_chunk_isal_oracle` returns Ok(false), gzip_chunk.rs:390-408). Candidate (a) "14/14 was oracle-forced" REFUTED (env-unset verified). Candidate (b) "window-seeding regressed 14→0" REFUTED (HEAD isal IS 14/14, no bisect needed). Cause = (c) mislabel. The "2% clean_flipped vs 14 coverage" paradox is resolved: clean_flipped counts only marker-loop pre-handoff clean bytes (gzip_chunk.rs:1900); the post-flip clean TAIL (the bulk) is decoded by ISA-L AFTER FlipToClean and is NOT in that counter. | plans/isal-dormancy-advisor-verdict.md; bins 2d317027 (isal) / a42d4600 (native) @ d56cb0f5 frozen guest; gzip_chunk.rs:390-408,669,1169,1185,1793-1818,1900 |
| DIS-14 | "The D1 8× ISA-L output OVER-RESERVE causes cache/TLB/page-fault pressure that slows the kernel (a per-byte-gap lever)." (capstone claim 3 suspect #2) | **REFUTED by causal perturbation (GZIPPY_ISAL_RESERVE_FACTOR A/B, frozen guest trainer, interleaved N=11, sha=OK, isal_chunks asserted, vs rg 0.16.0).** T1 (engine-matched ISA-L, no bootstrap — the clean per-byte cell): factor 8/12/16 → page-faults 56,946/56,947/56,944 (IDENTICAL TO THE BYTE) and wall 1025/1024/1024ms (Δ≤1ms), ratio_vs_rg invariant 0.903x. Lazy faulting touches only WRITTEN pages (== decoded bytes) regardless of capacity; the single reused buffer is sized once. T4: bigger reserve ADDS faults (105k→134k, rpmalloc multi-buffer span-cache spill) but only +7..18ms wall ⇒ faults are SLACK. Reserve is NOT freely reducible: factors 5/6/7 each force 1 pure-Rust fallback (one silesia chunk decodes ~7.5×), so **factor 8 is the tightest 0-fallback uniform factor AND already the wall+fault optimum**. The faithful incremental-growth fix has a CEILING of a few ms (T1 fault-neutral + T4 low fault→wall slope) — far below the ~50ms T4 / ~97ms T1 gaps. Also REFUTED (source): rg's avail_out feed is the WHOLE buffer (isal.hpp:258), same as gz — "rg refills in small chunks" is FALSE; and the inflate hot kernel is NASM (igzip_decode_block_stateless_0[14].asm) so cross-language LTO cannot touch per-byte rate (high-effort/zero-payoff, STOP). rg-uses-ISA-L premise VERIFIED (rg .so statically links decode_huffman_code_block_stateless_01/04). The residual ~0.097x@T1 is the PER-CHUNK/ParallelSM-PIPELINE term (architecture-port lever), NOT D1/glue. | branch perf/isal-d1-reserve @ dc14ba36; bin cebb7a43 @ HEAD frozen guest; gzip_chunk.rs isal_reserve_factor; scripts/bench/{d1.sh,_d1_guest.sh}; /tmp/d1_T1.log /tmp/d1_T4.log; orchestrator-status.md §D1 |
| DIS-15 | "isal low-T (~0.10 gap to rg) is a PROVED FLOOR (deeper than chunking — the in-process ISA-L call itself)." (the pre-registered FALSIFIER's null branch, DIS-14's residual owed its own oracle) | **REFUTED — and the OTHER branch CONFIRMED — by a single-shot removal oracle (GZIPPY_ISAL_SINGLESHOT=1 routes whole-stream through the EXISTING `isal_decompress::decompress_gzip_stream`: ONE ISA-L call, no chunking, no per-chunk set_dict, no ring/window-map/handoff). Frozen guest (bench-lock, no_turbo=1 gov=performance, runnable_avg=1.00), interleaved N=15, same-sink /dev/shm regular file, sha=028bd002…=OK every arm, vs rg 0.16.0; bin 9c466f67.** T1 (mask 0, spreads 0.6–1.0%): gz-prod ParallelSM-16chunk (FORCE_PARALLEL_SM=1, isal_chunks=16 fb=1) = 1.0131s / 209 MB/s / **0.905x** rg; **gz-singleshot = 0.7659s / 277 MB/s / 1.197x rg (BEATS rg by 20%)**; rg-file = 0.9164s / 231 MB/s. Removal-oracle pipeline cost = gz-prod − gz-singleshot = **247 ms (~24% of the T1 wall)**. So the ENTIRE ~0.10 T1 gap IS the per-chunk ParallelSM pipeline (pre-registered "singleshot~=rg ⇒ pipeline is the lever" — here it OVERSHOOTS), NOT a floor, NOT the ISA-L call (single-shot uses the SAME igzip kernel and is fastest). DECOMPOSITION (GZIPPY_VERBOSE counters @T1): markers are ZERO (flip_to_clean=0 finished_no_flip=0), window_seeded=16, finish_decode=17, inflate_wrapper=1, isal_fallbacks=1; the named "16× init+set_dict(32 KB)" component self-times **124 µs over 17 calls = 0.05% of the 247 ms** (negligible). ⇒ the 247 ms is chunk-LIFECYCLE: the 1 pure-Rust fallback re-decode + ring/window-map/CRC-per-chunk/handoff + the T1 SERIALIZATION (each chunk waits the prior chunk's 32 KB tail-window before it can ISA-L-decode → fully serial with handoff latency, zero parallelism benefit at T1). T4 (spreads 23–29%): gz-prod=0.5387s/0.911x rg, gz-singleshot=0.7543s/0.651x (single-threaded — can't use 4 cores, so the pipeline is net-POSITIVE +216 ms at T4: it is what BUYS parallelism). LEVER, REAL, SIZED: at T1 gzippy should route to single-shot ISA-L (a ROUTING fix, beats rg) rather than force chunking; the T4 0.911x residual is a separate parallel-scheduling gap. (Caveat for the gate: gz-prod is FORCE_PARALLEL_SM=1; whether real-production `-p1` forces ParallelSM is a routing question to confirm — but the campaign has measured T1 as forced-ParallelSM throughout, so the comparison is apples-to-apples with prior BAR-1 numbers.) | worktree .claude/worktrees/perchunk-singleshot, branch perf/perchunk-singleshot; bin 9c466f67 @ frozen guest; src/decompress/mod.rs `try_isal_singleshot_oracle` + `isal_decompress::decompress_gzip_stream` (existing single-shot); component timer in `decompress_deflate_from_bit_into` (isal_decompress.rs) printed by chunk_fetcher.rs; scripts/bench/{perchunk.sh,_perchunk_guest.sh}; orchestrator-status.md §PERCHUNK |
| DIS-16 | "The faithful CONSUMER-LEAN (removing the byte-transparent non-vendor consumer-top overheads D2/D3/D4 of pipeline-fidelity-verdict.md) is the BAR-1 lever for the T4 0.911x gzippy-ISAL gap that single-shot cannot probe." (the T4 sizing DIS-15 owed) | **NULL BRANCH — the lean is a TIE; the T4 gap is DEEPER parallel-scheduling, NOT the consumer-top per-chunk lifecycle. Refuted by a byte-transparent removal oracle (GZIPPY_LEAN_CONSUMER=1 skips D2 unconditional per-iter process_ready_prefetches+harvest chunk_fetcher.rs:1213/1218, D3 clean-branch full prefetch-cache scan :1715, D4 throwaway format! trace Strings :1578/1690/1697/1788/1814 — promotion/harvest still happen the SAME iteration inside block_fetcher.get/queue_prefetched/dispatch, so OFF==identity). Frozen guest (bench-lock no_turbo=1 gov=performance runnable_avg≈1.25, host QUIET), gzippy-isal feature, interleaved N=13 ×2 independent runs, same-sink /dev/shm regular file, sha=028bd002…=OK every arm (lean BYTE-TRANSPARENT at T1+T4), path=ParallelSM, isal_chunks=14/14 fb=0 asserted both arms, vs rg 0.16.0; bin 378788924ace0381; raw_bytes=211,968,000.** T4 mask 0,2,4,6: run1 gz-prod min=0.5470s/0.902x, gz-lean min=0.5524s/0.894x, rg-file min=0.4936s; run2 gz-prod min=0.5517s/0.892x, gz-lean min=0.5471s/0.899x, rg-file min=0.4921s. lean RECOVERY = gz-prod−gz-lean = **−5ms (run1) / +5ms (run2) — sign FLIPS, |Δ|≈5ms ≪ the 17–36% gzippy spreads (~90–200ms)** ⇒ TIE, recovers ~0. LIFECYCLE decomposition (GZIPPY_VERBOSE, the deliverable) shows WHY: the removed D2/D3/D4 are µs-scale — prefetch_promote 1→0 µs, window_publish ~0.5ms total, D4 format! sub-µs — i.e. 2–3 orders of magnitude below the ~55ms wall gap; the consumer self-time is dominated by fetcher_get (productive ISA-L decode-wait ≈115ms cumulative/17 iters) and postproc_dispatch (≈74ms), NEITHER touched by the lean. ⇒ The faithful consumer-lean is NOT the T4 BAR-1 lever; the T4 ~0.90x gap is in the productive parallel decode + post-process/scheduling path (head-of-line stalls / window-publish serialization), the owed parallel-scheduling oracle. (Scope: gzippy-NATIVE untouched, separate 0.667x engine floor.) | worktree .claude/worktrees/t4-lean-consumer, branch owner/t4-lean-consumer @ 1642cb58; bin 378788924ace0381 @ frozen guest; chunk_fetcher.rs `lean_consumer_enabled` + [LIFECYCLE] print; scripts/bench/{lean.sh,_lean_guest.sh}; orchestrator-status.md §LEAN-CONSUMER |

**SCOPE/STATUS ANNOTATIONS — Section B (2026-06-13)**

| id | SCOPE | LEDGER-STATUS | CONFIDENCE |
|----|-------|---------------|------------|
| DIS-1 | commit 690941f3, VAR_VII, neurotic guest, benches/engine_isolation.rs | **SCOPED** — VAR_VII-specific; the production engine has changed substantially; the general per-symbol-re-entry insight is sound but the specific 0.276x / 78 MB/s figures apply only to that variant | attribution + rate measurement (moderate) |
| DIS-2 | commit 64eb6df; oracle instrument fix | **HOLDS-AT-HEAD** — structural: the broken oracle was fixed; do not use the old clean-window oracle without that fix | structural/source-read |
| DIS-3 | chunk_fetcher.rs:870-874 grep string; script-only fix | **SCOPED** — if chunk_fetcher.rs counter names changed at HEAD, re-verify the coverage-assert string | source-read |
| DIS-4 | commits e4389f05/f216c691; neurotic host-freeze hardening | **HOLDS-AT-HEAD** — structural: HARD-FAIL on thawed box is enforced; the drift pattern is a box-level finding | instrument measurement |
| DIS-5 | neurotic guest, specific worktree/commit era, output-overlap | **SCOPED** — refutation holds in context; output routing code not re-verified at HEAD | removal-oracle (strong) within context |
| DIS-6 | d56cb0f5, gzip_block_finder.rs:180-182, chunk_fetcher.rs:1306/1431 | **STALE-RISK** — chunk_fetcher.rs is a heavily-modified file; verify the cited file:line references still show re-targeting before citing "gzippy ALREADY re-targets" as a current fact | source-read + empirical at d56cb0f5 |
| DIS-7 | commit b8a38e64; Oracle-C design flaw | **HOLDS-AT-HEAD** — structural: Oracle-C frees windows, collapsing the publish-chain; this is a design constraint independent of code changes | structural |
| DIS-8 | structural arithmetic (O(1) combine vs double-count) | **HOLDS-AT-HEAD** — structural: combine_crc is an O(1) combine of worker-computed CRCs; this property is independent of code changes | structural/source-read |
| DIS-9 | commit f98af1f, removal oracles, neurotic guest; C2 sub-terms | **SCOPED** — removal oracles at that commit; C2 sub-terms not re-measured at HEAD | removal-oracle (strong) within context |
| DIS-10 | isal.hpp:392-405 source-read + empirical (SYNC_FLUSH), d56cb0f5 | **SCOPED** — vendor isal.hpp is stable; the empirical boundary-recording claim scoped to d56cb0f5 | source-read (strong for structural claim); empirical (moderate for boundary-recording claim) |
| DIS-11 | GZIPPY_PERFECT_OVERLAP oracle; design flaw (upper bound by destruction) | **HOLDS-AT-HEAD** — structural: an upper-bound-by-destruction oracle cannot falsify a TIE-is-reachable claim; logic is code-independent | structural/logical |
| DIS-12 | d56cb0f5, build.rs:98-110, gzip_chunk.rs (:154,161,205,630,669,1169,1185), Cargo.toml:84 | **STALE-RISK** — src/decompress/mod.rs IS modified in the current working tree (git status M); gzip_chunk.rs line numbers and routing logic may have shifted; re-verify the ISA-L enable/disable chain at HEAD before citing | source-read at d56cb0f5 |
| DIS-13 | d56cb0f5, bins 2d317027/a42d4600, neurotic guest; mislabeled-binary class | **SCOPED** — the mislabeled-binary diagnosis is valid in context; gzip_chunk.rs:390-408 stub may have shifted line numbers at HEAD but the structural fact (native stub can never increment the isal_chunks counter) likely holds | source-read + empirical (strong for diagnosis mechanism) |
| DIS-14 | branch perf/isal-d1-reserve @ dc14ba36, bin cebb7a43, neurotic guest, silesia | **SCOPED** — causal perturbation (factor sweep, page-fault count), N=11 interleaved, sha=OK; CONFIDENCE: strong. NOTE: DIS-23 REFUTES DIS-14's source-only claim that "rg's avail_out feed is the WHOLE buffer, same as gz" — see DIS-23 annotation. The reserve-factor causal sweep itself is unaffected by that source correction | causal perturbation (strong) |
| DIS-15 | 2026-06-09, bin 9c466f67, neurotic guest 10.30.0.199 (no_turbo=1), gzippy-isal, silesia, T1 (FORCE_PARALLEL_SM=1); DIS-14 residual oracle | **SUPERSEDED-AT-HEAD (DISPUTED — see retraction below)** — INITIAL SUPERSESSION CLAIM: 2026-06-13 solvency N=10 sha-verified reported isal-PSM vs isal-shot T1 delta = ~6ms (not 247ms), suggesting the tax is gone; native T1 deficit appeared to be the INNER INFLATE KERNEL. RETRACTION (commit 39277bfb on branch fix/push-slice-batch-copy): the solvency "isal-PSM" arm used FORCE_PARALLEL_SM which is a **dead no-op (0 consumers in src/)** at HEAD — so both arms were effectively IsalSingleShot; the 6ms compared shot-vs-shot, NOT PSM-vs-shot. **The 247ms PSM T1 tax is NOT refuted; DIS-15 is SCOPED to d56cb0f5/neurotic/FORCE_PARALLEL_SM-operative context.** Mechanism update: at HEAD, gzippy-isal T1 single-member BYPASSES PSM via the IsalSingleShot route (DIS-22; mod.rs contains DecodePath::IsalSingleShot), so the 247ms tax is **moot for production** (PSM is never taken at T1). The native T1 split (pipeline vs kernel) is UNKNOWN pending a native single-shot oracle. CONFIDENCE of original 247ms finding: removal-oracle (strong) for its context; CONFIDENCE of the 6ms "gone" claim: VOID (retracted). ORIGINAL LANGUAGE NOTE: "PROVED FLOOR" in the claim column was the prior claim being disproved by DIS-15, not a claim DIS-15 itself makes; that disproof was correct in its context and remains so. | SUPERSEDED-AT-HEAD (DISPUTED) — see retraction in annotation |
| DIS-16 | worktree .claude/worktrees/t4-lean-consumer @ 1642cb58, bin 378788924ace0381, neurotic guest, silesia, T4 | **SCOPED** — TIE verdict on a specific binary/date; if chunk_fetcher.rs:1213/1218/1715 regions have changed at HEAD, re-verify before concluding lean-consumer is not the T4 lever | removal-oracle (strong) within context |

---

## C. RE-MEASURED TIEs — STOP re-measuring (binding while the cited region is unchanged)

A correct (byte-identical) change is KEPT/layered even on a TIE (rule 7a). These
are TIE *verdicts on those runs* — NOT refutations of the direction — but the
direction has been measured to a TIE more than once and must NOT be re-measured
without a NEW mechanism (rule 7).

| id | direction | TIE evidence (>=2 measurements) | pin |
|----|-----------|--------------------------------|-----|
| TIE-1 | Marker fast-loop (rg multi-cached u16 marker loop port). | markerfast vs mergefix +1.2%/+3.0%/+0.0% = TIE; ported, byte-exact, KEPT 7a. Marker loop is FASTER than rg already => not the binder. | commit 04fda86d; plans/marker-loop-port-advisor-verdict.md |
| TIE-2 | Inner-Huffman STORE-side fastloop2 (ring VAR_V speculative packed). | 1.001x/1.018x/0.994x = TIE; KEPT 7a. | commit 2ff19ac6; plans/correctness-net-advisor-verdict.md |
| TIE-3 | Finer chunking (GZIPPY_CHUNK_KIB sweep). | REFUTED ×2 runs, byte-exact, interleaved vs rg. | orchestrator-status :805-812 |
| TIE-4 | writev granularity (GZIPPY_WRITEV_CAP_KIB {2048,256,95}). | all TIE/worse => lever is write TIMING not size. | orchestrator-status :411 |
| TIE-5 | Engine swap pure-Rust->ISA-L when SEEDED at T8 (window-absent removed). | `pure` (slower engine) ALREADY ties rg seeded; engine swap TIE-vs-TIE => engine NOT the T8 binder (T8 is slack-masked). | commit a884fa7b; plans/clean-rate-ceiling-advisor-verdict.md; orchestrator-status :1089-1092 |
| TIE-6 | Marker-segment page-warmth sub-lever. | REFUTED (−12% fault ceiling; underpowered TIE, max plausible win 5-10x below spread). Advisor ×3. | commit f80294ae; plans/page-warmth-rootcause-advisor-verdict.md |

NOTE on TIE-5: this is a T8-ONLY tie under window-absent-removal. It does NOT
extend to T4/T1 — at T4 the engine swap moves the wall 0.740->0.899 (>>spread, see
LEV-1). "Engine is slack-masked" is a T8 statement; do not generalize it to low-T.

**SCOPE/STATUS ANNOTATIONS — Section C (2026-06-13)**

| id | SCOPE | LEDGER-STATUS | CONFIDENCE |
|----|-------|---------------|------------|
| TIE-1 | commit 04fda86d, neurotic guest, silesia; marker fast-loop | **SCOPED** — TIE verdict on that binary; re-measure if marker loop is further modified | 3 runs, sign-stable; moderate |
| TIE-2 | commit 2ff19ac6, neurotic guest; inner-Huffman fastloop2 | **SCOPED** | 3 runs; moderate |
| TIE-3 | 2 runs, neurotic guest; finer chunking sweep | **SCOPED** — re-measure if chunk-sizing logic changes | 2 runs; weak |
| TIE-4 | neurotic guest; writev granularity sweep | **SCOPED** | sweep; moderate |
| TIE-5 | commit a884fa7b, neurotic guest, silesia, **T8 ONLY** (noted in table + below-table caveat) | **SCOPED** — T8-ONLY scope explicitly stated; do NOT generalize to T1/T4. At T4 the engine swap is a real lever (LEV-1: 0.740→0.899). The "engine is slack-masked" label is only valid in the T8/window-absent-removal context | removal-oracle-grade at T8 (strong); scope-limited |
| TIE-6 | commit f80294ae, neurotic guest; page-warmth sub-lever | **SCOPED** — underpowered TIE; low effect ceiling sized | causal perturbation; moderate |

---

## D. LIVE "THE LEVER IS X" LINES — perturbation-confirmed vs ATTRIBUTION-ONLY

| id | lever claim | confirmation status | pin |
|----|-------------|---------------------|-----|
| LEV-1 | **Engine swap (pure-Rust->ISA-L) recovers ~0.159x of native's T4 deficit.** | **CONFIRMED by REMOVAL ORACLE** (ocl_cf ISA-L engine-isolation = a real engine swap, not attribution): native 0.740x -> ocl_cf 0.899x, delta 0.159x ≈ 5× spread, sign-stable, both TIGHT (<=5%). This is a causal swap, not producer-side blame. | falsifier MEASURED block; orchestrator-status JOB 1; bins b9eb0a73 / 710a6dc @ d56cb0f5 |
| LEV-2 | **Non-engine residual ~55ms / <=0.101x survives even ISA-L at T4.** | **UN-STALED (2026-06-08 dormancy-reconciliation): the "STALE-AT-HEAD" downgrade rested on the mislabeled-native dormancy number (DIS-13) and is REVERSED.** At HEAD the env-unset isal binary IS 0.900x at T4 (ocl_cf reproduces), so the 0.159/0.101 partition is NOT stale. BUT LEV-2 remains **UNCONFIRMED as a TURNABLE LEVER — it is an UPPER BOUND, not a localized binder.** ocl_cf/isal is a BLEND (source-verified, gzip_chunk.rs: ISA-L covers the post-32KiB-flip clean tail continuation; the <=32KiB markered prefix + chunk-0 bootstrap STAY pure-Rust). So 0.101x BUNDLES (i) marker-prefix pure-Rust engine compute, (ii) per-chunk FFI/handoff, (iii) scheduling/bootstrap. NO causal perturbation has decomposed it (OPEN-1 owed: ISA-L-the-marker-prefix oracle OR an FFI-null run). Cannot be turned until decomposed. NOTE the prior "clean-only placement is a 10ms TIE ⇒ residual within-spread" oracle (OPEN-2) was ALSO run on a build of unverified feature; it should be re-validated on the verified isal binary before banking placement-is-slack. | low-t-gate-advisor-verdict.md; plans/isal-dormancy-advisor-verdict.md; gzip_chunk.rs (full-window gate) VERIFIED @ d56cb0f5 |
| LEV-3 | "Clean-mode inner-Huffman COMPUTE is ON the T8 critical path." | **CONFIRMED (causal perturbation): slow-injection slope monotonic + SURVIVES frequency-neutral sleep control.** BUT its SHARE is only ~11-29% (methods disagree) => bounded; even infinite clean speedup lands 0.79-1.00s, still loses. On-path ≠ dominant. | memory project_pregate_placement (pre-gate); HEAD d0aa1db |
| LEV-4 | "Engine clean-rate is ~2.3x slower than rapidgzip and survives perfect placement." | **CONFIRMED (clean-only removal oracle, publish-chain preserved, byte-exact, self-test passed): engine ceiling 0.6134s vs rg 0.5396s = +13.7% (3.47σ, NOT a tie); per-chunk 92.7ms vs 39ms = 2.38x.** Conservative lower bound. | memory project_pregate_placement; commit b8a38e64 |
| LEV-5 | "Placement / head-of-line stalls is the dominant T8 lever (~58% lost to serialization)." | **PARTIALLY confirmed, RE-SCOPED.** Oracle-P (placement-perfect) = 0.56-0.66s vs rg 0.524s = still 7-26% LOSS => placement is NECESSARY-but-INSUFFICIENT, co-primary with engine (NOT "the dominant lever"). The original "dominant single lever" framing was REFUTED. | memory project_pregate_placement §CORRECTION; commit c829982 |
| LEV-6 | "Consumer non-decode SERIAL floor is ~225ms (forbids the tie)." | **REFUTED by perturbation/decomposition: real serial bookkeeping = ~15ms; the ~225ms was decode-WAIT mis-bucketed as serial** (decode-wait ≈ 0.49s = 97%). Tie NOT structurally forbidden by the consumer. CAVEAT: trace was /dev/null; same-sink file-write adds ~0.245s. | memory project_pregate_placement STEP-0; commit 598e22a4 |
| LEV-7 | "Confirmed-offset prefetch gap (head-of-line stalls) is the non-engine binder, ~40% of T8 wall." | **ATTRIBUTION + located, but the FIX is REFUTED** (see DIS-6). The stalls are real & measured (4× decode_NOT_STARTED, all consumer-pace/never-retained), but consumer-confirmation prefetch CANNOT close them and offset-supply is a no-op. The OPEN sub-question (worker saturation vs shallow prefetch horizon) is UNRESOLVED. So: the gap EXISTS (measured) but is NOT yet a turnable lever. | memory project_confirmed_offset_prefetch_gap |
| LEV-8 | "merge-removal (view-based applyWindow port) moves T8 +12% (0.65x->0.73x)." | **CONFIRMED (landed, byte-exact, advisor-UPHELD, sign-stable wall move).** A faithful port that actually moved the wall. KEPT. | commit (merge-removal); orchestrator-status :943; plans/merge-removal-advisor-verdict.md |
| LEV-9 | "copy-free FOLD clean drain recovers the cadence tax (+0.059x, native 0.678->0.737x)." | **CONFIRMED (cadence/intrinsic split, advisor's owed symmetric control ×2; freq-neutral; byte-exact). KEPT ratchet tooth.** | commit 9cde0b4f; orchestrator-status :601-607 |

**SCOPE/STATUS ANNOTATIONS — Section D (2026-06-13)**

| id | SCOPE | LEDGER-STATUS | CONFIDENCE |
|----|-------|---------------|------------|
| LEV-1 | 2026-06-08, d56cb0f5, bins b9eb0a73/710a6dc, neurotic guest, silesia, T4 | **SCOPED** — removal-oracle-grade (real ISA-L engine swap, two production binaries); strong within context. Re-verify if engine routing or the gzippy-isal/native feature distinction changes | removal-oracle (strong) |
| LEV-2 | 2026-06-08, d56cb0f5, gzippy-isal env-unset, neurotic guest, silesia, T4 | **SCOPED** — already labeled "UNCONFIRMED as a TURNABLE LEVER"; the 0.101x is an UPPER BOUND that bundles marker-prefix engine + FFI/handoff + scheduling. CONFIDENCE: strong for the 0.101x upper bound arithmetic; moderate for any lever-size inference from it | measurement (strong for bound); weak for lever authorization |
| LEV-3 | HEAD d0aa1db, causal perturbation + freq-neutral sleep control, T8 | **SCOPED** — T8-specific. The "~11-29% share" range reflects method disagreement; cite the range, not a point estimate. CONFIDENCE: moderate (monotonic slope survives sleep swap, but share estimate is method-dependent) | causal perturbation + freq-neutral; moderate |
| LEV-4 | commit b8a38e64, clean-only removal oracle, publish-chain preserved | **SCOPED** — removal oracle, 3.47σ; strong within context. NOTE: DIS-19 shows rg ALSO decodes all chunks via its marker engine (not u8-direct ISA-L for streaming); the "2.3x per-chunk rate" comparison is between gzippy's marker engine and rg's marker engine, both paying marker cost — the gap retains directional validity | removal-oracle (strong) |
| LEV-5 | commit c829982, Oracle-P; already labeled "PARTIALLY confirmed, RE-SCOPED" | **SCOPED** — "dominant single lever" framing refuted; placement is necessary-but-insufficient, co-primary with engine | removal-oracle-grade within context; moderate |
| LEV-6 | commit 598e22a4, /dev/null trace; serial bookkeeping = ~15ms | **SCOPED** — /dev/null caveat already noted (same-sink file-write adds ~0.245s); re-verify if cited for same-sink consumer floor | perturbation + decomposition; moderate |
| LEV-7 | memory + commits; gap located, fix refuted (DIS-6) | **SCOPED** — gap is measured; the fix remains blocked by DIS-6's mechanism (DIS-6 itself is STALE-RISK — see DIS-6 annotation). CONFIDENCE: moderate for gap existence; strong for the fix-refutation in its context | measurement (moderate); fix-refutation (strong in context) |
| LEV-8 | commit (merge-removal), neurotic guest; +12% wall move | **SCOPED** — confirmed landed, byte-exact, advisor-upheld. CONFIDENCE: strong (signed wall move, stable, advisor ×1) | signed wall move; strong |
| LEV-9 | commit 9cde0b4f, freq-neutral, byte-exact; +0.059x native | **SCOPED** — confirmed ratchet tooth, freq-neutral, advisor ×2. CONFIDENCE: strong | causal perturbation + freq-neutral (strong) |

---

## E. OPEN FALSIFIERS (pre-registered, not yet resolved)

| id | falsifier | state | pin |
|----|-----------|-------|-----|
| OPEN-1 | F-NON-ENGINE residual decomposition: separate the 0.101x T4 residual into marker-prefix-engine / FFI-handoff / scheduling buckets (advisor-owed oracle: ISA-L-the-marker-prefix OR FFI-null run). Until then LEV-2 is not turnable. | OPEN — the next owner deliverable (residual-decomposition oracle). | low-t-gate-advisor-verdict.md :17; standing-specialists.md §NEXT OWNER LEVER |
| OPEN-2 | The confirmed-offset OPEN sub-question: are the T4/T8 decode_NOT_STARTED stalls WORKER SATURATION (engine) or PREFETCH-HORIZON too shallow (structural)? Decisive diagnostic not yet run; do NOT conclude "all engine". | **RE-OPENED for the load-bearing leg (2026-06-08 dormancy-reconciliation).** The "clean-only placement is a 10ms TIE ⇒ placement slack" oracle that bounded OPEN-2 was run alongside the mislabeled-native dormancy number (its baseline "production same-sink 654ms=0.757x" is the NATIVE wall, DIS-13, NOT gzippy-isal at 0.900x). The clean-only-vs-production comparison must be RE-RUN on the VERIFIED isal binary (0.900x baseline) before banking "placement recovers ~0." The "98% marker work" reason is also a misread (the post-flip clean tail offloads to ISA-L on the isal build; it is NOT all marker work). Until re-validated on the verified binary, OPEN-2 is UNRESOLVED, not closed. | plans/residual-attribution-advisor-verdict.md; plans/isal-dormancy-advisor-verdict.md; OWED re-run on bin 2d317027 |
| OPEN-3 | JOB 2 stored/fixed clean-tail coverage: relax until_exact EXACT-match accept to coalesce to nearest clean EOB (faithful readStream). Gap is REAL on SYNC_FLUSH-dense input; fix is a correctness-sensitive seed-path change warranting its OWN gated turn (risks re-introducing 19add96c over-decode mis-seed). | OPEN — gated. | orchestrator-status JOB 2; gzip_chunk.rs:302-333 |
| OPEN-4 | Engine plateau falsifier (E2 SIMD back-ref / E3 packed multi-literal / E4 wide refill): defined over variant (ii) AFTER E2-E4 are built. E1 (output width, +6%) was only output => plateau NOT-PROVEN-YET. | OPEN | memory project_pregate_placement §ENGINE ISOLATION; base 249f25b5 |
| OPEN-5 | Steward retro-validation of the JUST-USED T4 gate (0.899x / 0.740x). If contaminated, the whole JOB-1 PARTIAL verdict is void. (Steward's deliverable; the Red-team ledger TRAILS it.) | **RE-CLOSED IN FAVOR OF THE BANKED GATE (2026-06-08, dormancy-reconciliation owner).** The OPEN-5 "FIRED" finding rested on the engine-isolation `--kind` abort + the fresh isal_chunks=0 — both now traced to a **mislabeled NATIVE binary** (DIS-13). At HEAD d56cb0f5 the env-unset isal binary measures isal_chunks=14/14 fallbacks=0 (first-hand, frozen, N=11, sha=OK) — the banked 0.899x ocl_cf gate REPRODUCES (env-unset isal T4 = 0.900x ≈ ocl_cf 0.899x; native T4 = 0.761x ≈ banked 0.740x). The `--kind engine-isolation` abort was a SCRIPT/binary issue on a native build, not an ISA-L-dormancy fact. LEV-1 (engine swap recovers ~0.159x at T4) is RE-CONFIRMED by the env-unset isal-vs-native A/B (0.900x − 0.761x = 0.139x, same order, removal-oracle-grade since it's two real production binaries). LEV-2 (the <=0.101x non-engine residual) is NOT stale-at-HEAD after all — but remains UNCONFIRMED as a turnable lever (it bundles marker-prefix engine + FFI/handoff + scheduling; OPEN-1 decomposition still owed). | plans/isal-dormancy-advisor-verdict.md; bins 2d317027/a42d4600 @ d56cb0f5 frozen guest |

**SCOPE/STATUS ANNOTATIONS — Section E (2026-06-13)**

| id | SCOPE | LEDGER-STATUS |
|----|-------|---------------|
| OPEN-1 | 2026-06-08, d56cb0f5, neurotic guest; re-opened by dormancy-reconciliation; owed on bin 2d317027 | **SCOPED** — the seed-all 231ms ceiling is an over-removal (uncounted p=1 pre-pass; rg also pays marker cost per DIS-19); the Opus disproof gate was not run at time of seeding; CONFIDENCE: moderate (causal perturbation + T1 self-test PASS, but seed-all over-removes) |
| OPEN-2 | 2026-06-08, d56cb0f5; re-opened (mislabeled-native baseline invalidated prior "clean-only TIE" oracle) | **SCOPED** — target re-run: verified isal bin 2d317027 @ d56cb0f5 on neurotic |
| OPEN-3 | d56cb0f5, gzip_chunk.rs:302-333; correctness-sensitive gated turn | **SCOPED** — verify gzip_chunk.rs:302-333 line numbers at HEAD before attempting (single_member.rs IS modified in git status) |
| OPEN-4 | base 249f25b5, VAR_VIII bench; E2/E3/E4 variants not yet built | **SCOPED** — VAR_VIII was measured (see prose section); E2/E3/E4 remain open |
| OPEN-5 | 2026-06-08, d56cb0f5; re-closed by dormancy-reconciliation | **SCOPED** — re-closed; the 0.899x ocl_cf gate reproduced on verified isal bin at d56cb0f5 |

---

## F. INSTRUMENT DISTRUST REGISTRY (instruments proven broken — distrust until self-tested)

| id | instrument | failure | pin |
|----|------------|---------|-----|
| INSTR-1 | clean-window oracle | silently re-ran the full bootstrap (DIS-2). | commit 64eb6df |
| INSTR-2 | a second oracle | emitted EMPTY output. | CLAUDE.md rule 4 |
| INSTR-3 | isal_oracle_chunks= coverage grep | grepped a string the binary never emits (DIS-3); hard-failed silently. | chunk_fetcher.rs:870-874 |
| INSTR-4 | Oracle-C (free-decode) | degenerate — frees windows, collapses publish-chain (DIS-7). | commit b8a38e64 |
| INSTR-5 | GZIPPY_PERFECT_OVERLAP | built backwards (destroys overlap; can't falsify the TIE) (DIS-11). | plans/perfect-overlap-advisor-verdict.md |
| INSTR-6 | host-freeze WARN-only | let a THAWED box (no_turbo=0) drift ocl_cf 0.945↔0.989 (DIS-4); now HARD-FAIL. | commit f216c691 |
| INSTR-7 | rss_vs_t.sh MPKI label | awk divisor `-nan` under paranoid=4 (RAW counters valid; label only). | orchestrator-status :124 |

**SCOPE/STATUS ANNOTATIONS — Section F (2026-06-13)**

| id | SCOPE | LEDGER-STATUS |
|----|-------|---------------|
| INSTR-1 | commit 64eb6df; clean-window oracle fix | **HOLDS-AT-HEAD** — structural: do not use the old clean-window oracle without the commit-64eb6df fix |
| INSTR-2 | structural design defect (empty output) | **HOLDS-AT-HEAD** — structural: always self-test new oracles with non-empty, sha-verified output |
| INSTR-3 | chunk_fetcher.rs:870-874; script-only grep fix | **SCOPED** — if chunk_fetcher.rs counter names changed at HEAD, the coverage-assert string must be updated; verify before using |
| INSTR-4 | commit b8a38e64; Oracle-C design flaw | **HOLDS-AT-HEAD** — structural: free-decode collapses the publish-chain; the design constraint is code-independent |
| INSTR-5 | GZIPPY_PERFECT_OVERLAP; oracle design flaw | **HOLDS-AT-HEAD** — structural: upper-bound-by-destruction cannot falsify TIE-is-reachable; logic is code-independent |
| INSTR-6 | commit f216c691; now HARD-FAIL on thawed box | **HOLDS-AT-HEAD** — structural: HARD-FAIL enforcement is in place; always verify no_turbo=1 before measurements |
| INSTR-7 | orchestrator-status:124; awk label-only bug under paranoid=4 | **SCOPED** — label-only; RAW counters valid. Re-check if rss_vs_t.sh is still in active use |

---

## G. RED-TEAM RULING ON THE FRESH GATE INFERENCE

> **LEDGER ANNOTATION 2026-06-13** | SCOPE: 2026-06-08, d56cb0f5, neurotic guest, bins b9eb0a73/710a6dc, silesia, T4 | STATUS: **SCOPED** | CONFIDENCE: source-verified reasoning at d56cb0f5; the 0.159/0.101 partition arithmetic is sound but "non-engine residual" is already flagged in the text as a MISNOMER (it bundles marker-prefix engine + FFI/handoff + scheduling). The T1 picture has since changed (DIS-15 SUPERSEDED-AT-HEAD); this ruling applies to the T4 regime only and has not been re-verified on solvency at HEAD.

CLAIM under review: "ocl_cf-T4 0.899x / native-T4 0.740x => even a perfect engine
loses at T4; the non-engine residual <=0.101x must close."

**Ruling: the FIRST half is SOUND; the SECOND half OVER-CLAIMS and must not be
banked as a sized lever.**

1. SOUND: "even REAL ISA-L (the fastest engine) loses 0.899x at T4" — this is a
   removal-oracle result (LEV-1), TIGHT (3-4% spread), and 0.899 sits ~2.5× spread
   below 0.99. The engine ALONE does not reach parity at T4. F-ENGINE-CLOSABLE is
   correctly rejected. asm's best case == ISA-L == 0.899 (zero-margin) is correctly
   flagged. No objection.

2. OVER-CLAIM / INFERENCE RISK — the 0.101x is NOT a clean non-engine number:
   - **It double-bundles, it does NOT double-count.** 0.159x (engine share) and
     0.101x (residual) are COMPLEMENTARY shares of the SAME 0.260x native-vs-rg T4
     gap (1.0 − 0.740 = 0.260; 0.159 + 0.101 = 0.260). They PARTITION the deficit,
     so they do not double-count the marker prefix between EACH OTHER. The risk is
     the OPPOSITE: 0.101x is an UPPER BOUND that ITSELF still contains engine work.
   - **Source-verified at HEAD d56cb0f5** (gzip_chunk.rs `finish_decode_chunk_isal_oracle`,
     full-32KiB-window gate): the ISA-L oracle routes ONLY the clean full-window
     continuation through FFI; the <=32KiB markered prefix + chunk-0 bootstrap STAY
     in the pure-Rust u16 marker engine. So ocl_cf=0.899 STILL pays pure-Rust engine
     time on the prefix, and 0.101x bundles {marker-prefix engine + per-chunk FFI/
     handoff + scheduling}. "Non-engine residual" is a MISNOMER for it.
   - **Consequence for the claim:** "the non-engine residual <=0.101x must close" is
     true as an arithmetic identity (to reach 1.0 from 0.899 you must also close the
     part the engine swap didn't), but FALSE as "<=0.101x of pure scheduling must
     close." Part of that 0.101x is more ENGINE (the marker prefix), which an asm
     engine WOULD capture — so the genuinely-non-engine (scheduling/bootstrap) slice
     is SMALLER than 0.101x and is currently UNSIZED. Treating 0.101x as the
     scheduling-lever budget over-sizes that lever.
   - **Additivity check (the prompt's question):** YES, 0.159 + 0.101 are additive
     (they sum to the 0.260 deficit). They do NOT double-count the marker prefix
     between themselves. The marker prefix lives ENTIRELY inside the 0.101x bucket
     (it is the part ISA-L did NOT touch). So the risk is mis-LABELLING within the
     0.101x bucket, not double-counting across the two buckets.

3. STANDING INFERENCE RISK: do NOT let "even a perfect engine loses at T4 => the
   residual is non-engine => attack scheduling" become a lever authorization.
   That chain has an UNCONFIRMED middle term (LEV-2). The owed decomposition oracle
   (OPEN-1) is the gate; until it runs, the scheduling lever is attribution-only.
</content>
</invoke>

---

## OPEN-1 DECOMPOSITION RUN — the owed low-T residual split, on the VERIFIED isal binary [2026-06-08, owner/lowt-residual-decomp @ d56cb0f5, bin b9eb0a73]

> **LEDGER ANNOTATION 2026-06-13** | SCOPE: 2026-06-08, owner/lowt-residual-decomp @ d56cb0f5, bin b9eb0a73, neurotic guest 10.30.0.199 (no_turbo=1), gzippy-isal, silesia, T4/T1 | STATUS: **SCOPED** | CONFIDENCE: causal perturbation (seed-all removal, T1 self-test PASS, no-windows/no-bounds decompose). CAVEAT: the seed-all 231ms/1.55x ceiling is an OVER-REMOVAL (uncounted p=1 pre-pass, work rg also does at runtime); it is an upper bound, not a reachable target. The Opus disproof gate was not run at time of writing ("OWES"). The T1 analysis ("T1 is engine/FFI/output floor") is partly superseded by DIS-15's SUPERSEDED-AT-HEAD annotation — re-read T1 conclusions against the 2026-06-13 solvency finding.

STEP-0 PROOF-OF-BINARY (the prior turn's VOID failure mode eliminated): freshly built
gzippy-isal at HEAD, env-unset GZIPPY_VERBOSE: **isal_chunks=14/14/16 (T4/T8/T1),
fallbacks=0/0/1, path=ParallelSM, GZIPPY_ISAL_ENGINE_ORACLE+SEED UNSET.** isal_chunks>0
is structurally impossible on native (stub gzip_chunk.rs:390-400 returns Ok(false), never
increments :386). This IS the real-ISA-L binary. Frozen guest 10.30.0.199 (driver --lock,
runnable_avg 1.00-1.75 <=2.0, no_turbo=1), interleaved N=11, same-sink /dev/shm regular file.

NUMBERS (T4, mask 0,2,4,6, vs rg 0.16.0):
- production (same-sink, env-unset): 549ms = **0.902-0.904x** (sha=OK, spread 3-4%, x2 reproduced).
- seed-all clean-only (REAL capture+replay, hits=16 misses=0 flip_to_clean=0 window_seeded=17
  isal_chunks=17, OUTPUT SHA-VERIFIED 028bd002...cb410f byte-perfect 211968000): 318-325ms =
  **1.486-1.555x** (spread 6-11%). Removing the marker bootstrap (A marker-prefix + C placement +
  the resolve pass) SAVES ~225-231ms at T4.
- DECOMPOSE split (boundaries vs windows):
  - no-windows (boundaries pre-seeded, marker COMPUTE still paid): 581ms = 0.855x — NO help (worse).
  - no-bounds  (correct windows, partition-GUESS offsets):        555ms = 0.891x — ~= production.
  => the 231ms benefit needs BOTH; neither alone moves the wall. A=marker-compute is the cost,
     C=alignment is the necessary enabler. CO-PRIMARY / entangled (matches
     project_pregate_placement_is_dominant_lever).

INSTRUMENT-VALIDITY CATCH (rule 4): oracle.sh `--kind clean-only` sets GZIPPY_SEED_WINDOWS=1,
but that env is a REPLAY-FILE path, not a boolean — "=1" => open("1") ENOENT => silent fallback
to PRODUCTION (replay hits=0 misses=16). So `--kind clean-only` is a NO-OP on this codebase and
any "clean-only TIE/+10ms" banked from it (incl. the prior void-native OPEN-2) measured
PRODUCTION-vs-PRODUCTION. The REAL oracle is two-pass: GZIPPY_SEED_WINDOWS_CAPTURE=<f> at p=1,
then GZIPPY_SEED_WINDOWS=<f>. Drivers: scripts/bench/{_seed_clean_guest,_seed_decompose_guest,
seed_clean,seed_decompose}.sh (worktree). [HARNESS BUG — flag to Steward; oracle.sh clean-only
needs the capture step or it lies.]

T1 (mask 0): seed is a NO-OP (sequential => windows always present => clean path already taken,
inflate_wrapper=16, seed hits=0). So at T1 there is NO marker bootstrap to remove; T1 0.899x is
the ENGINE symbol-rate + per-chunk FFI-handoff + serial output floor — bounded by REAL ISA-L
itself (banked ocl_cf 0.899x zero-margin). The low-T residual is THREAD-COUNT-STRUCTURED:
  - T1: engine/FFI/output floor (no bootstrap). NOT closable by removing bootstrap.
  - T4: ADDS the marker-bootstrap (A+C), a LARGE removable term (231ms ceiling) that production
    leaves on the table because gzippy's u16 bootstrap is slower-per-byte than rg's AND placement
    is imperfect.

SPLIT VERDICT {marker-prefix(A) / FFI(B) / scheduling(C)} at T4:
- DOMINANT = (A) marker-prefix/bootstrap COMPUTE, entangled with (C) alignment (co-primary). It
  is the whole 231ms seed-all ceiling. (A) is the pure-Rust u16 marker decode+resolve of the
  window-absent prefix.
- (B) FFI-handoff is NOT a significant residual: the 318ms seed-all run still routes ALL 17 chunks
  through ISA-L FFI (isal_chunks=17) yet BEATS rg — if FFI-handoff were a big term seed-all could
  not be 1.5x rg. Bounded small.
- (C) alignment alone = 0 (no-windows oracle no help); it only matters as A's enabler.

FAITHFULNESS / CLOSABILITY (the load-bearing caveat — rule 3 + masks-binder):
- seed-all is a masks-binder CEILING that OVER-removes: it hands gzippy precomputed correct
  predecessor windows from an uncounted p=1 pre-pass, work rg ALSO does at runtime (rg carries the
  SAME u16 marker machinery, 31-34% replaced markers per CLAUDE.md). So 1.55x is an UPPER BOUND on
  the bootstrap lever, NOT a reachable target. The FAITHFUL gap is: gzippy's marker-bootstrap is
  SLOWER per byte than rg's window-map + in-place narrowing, AND gzippy's placement is less precise
  than rg's per-chunk window map. The faithfully-closable mechanism = rg's window-map/marker-resolve
  pipeline (DecodedData.hpp in-place narrowing + the per-chunk window map), NOT a new divergent path.
- This does NOT establish T4 reaches >=0.99x by closing the bootstrap — it establishes the bootstrap
  is the DOMINANT low-T-at-T4 term and FFI is not. The engine floor (T1 0.899x) still binds.

OWES: synchronous Opus disproof + Steward bankability sign-off (Agent tool absent in owner env =>
self-disproof only). The seed-all 1.55x over-removal must be discounted by an Opus pass before any
lever authorization.

## OPEN-1 DISENTANGLEMENT — the faithful-placement slice, ENGINE HELD (isal_chunks<=14 asserted) [2026-06-08, owner/disentangle-placement @ d56cb0f5, bin b9eb0a73]

> **LEDGER ANNOTATION 2026-06-13** | SCOPE: 2026-06-08, owner/disentangle-placement @ d56cb0f5, bin b9eb0a73, neurotic guest 10.30.0.199 (no_turbo=1), gzippy-isal, silesia, T4 | STATUS: **SCOPED** | CONFIDENCE: causal perturbation (placement-marker oracle, T1 self-test PASS, pre-registered falsifier fired). The CONCLUSION — "T4 low-T lever is the GATED ISA-L engine swap, NOT faithful runtime placement" — is backed by a pre-registered falsifier result (placement-marker was TIE-or-worse). The Opus disproof gate was not run at time of writing ("OWES"). Re-verify if isal_chunks routing or the placement oracle code changes.

This RESOLVES the confound the advisor (lowt-residual-gate-verdict) flagged: the seed-all 231ms
bundled (1) a gated pure-Rust->ISA-L engine swap on ~3 prefix chunks (isal_chunks 14->17), (2) a
free precomputed p=1 pre-pass placement, (3) the genuinely-faithful runtime-placement slice. This
turn SIZES (3) alone with a NEW byte-transparent oracle (`oracle.sh --kind placement-marker`):
seed the REAL boundaries (inline p=1 capture) but `GZIPPY_SEED_NO_WINDOWS=1` SUPPRESSES the
seeded-window clean upgrade => the window-absent prefix STAYS on the pure-Rust marker engine.

STEP-0 PROOF-OF-BINARY: freshly built gzippy-isal at HEAD (feature gzippy-isal, x86_64, RUSTFLAGS
target-cpu=native), env-unset GZIPPY_VERBOSE: **isal_chunks=14/14/16 (T4/T8/T1), fallbacks=0/0/1,
path=ParallelSM**, corpus sha 028bd002… == pin. Real-ISA-L binary (native stub :390-400 can never
increment :386). bin_sha b9eb0a73.

INVARIANT READBACK (the engine-leak gate): placement-marker T4 **isal_chunks=12-13 (3 runs:
13/12/13), fallbacks=1, seed hits=0 misses=16** — isal_chunks did NOT RISE above production's 14
(it DROPPED: one clean-tail chunk fell BACK to pure-Rust). NO gated-engine leak; engine held or
MORE-conservative. seed hits=0 proves the window-clean upgrade was fully suppressed (only the
boundary seed is active). Output SHA-VERIFIED byte-perfect (028bd002…cb410f).

NUMBERS (frozen guest 10.30.0.199, bench-lock host-freeze, quiet-gate runnable_avg<=2.0, no_turbo=1,
governor=performance, interleaved N=11, same-sink /dev/shm regular file, vs rg 0.16.0, mask 0,2,4,6):
- T4 production (same-sink, env-unset): **546ms = 0.906x** (rg 494ms, spread gz 4% / rg 3%, sha=OK).
- T4 placement-marker (boundaries seeded, ENGINE HELD): **580ms = 0.845x** (rg 490ms, spread gz 8%,
  sha=OK, isal_chunks<=14 asserted).
- T1 control (mask 0): production 1022ms = 0.898x; placement-marker 1023ms = 0.897x (seed hits=0
  misses=0, isal_chunks=16) — **IDENTITY at T1** (seed is a no-op when windows are always present).
  Instrument self-test PASSES: where there is no marker bootstrap (T1), the oracle == production.

FALSIFIER VERDICT (pre-registered in open1-disentangle-placement.md): the faithful-placement slice
= production_wall − oracle_wall = 546 − 580 = **−34ms (placement is TIE-to-slightly-WORSE)**, NOT a
positive lever. Δ is within combined inter-run spread (prod 4%≈22ms, oracle 8%≈46ms). Per the
falsifier: slice <= spread (and here NEGATIVE) ⇒ **placement is NOT a faithful low-T lever at T4.**
The oracle's 580ms independently REPRODUCES the prior owner's no-windows leg (581ms) on a fresh
binary/session. a-fortiori note: this oracle granted FREE precomputed placement (uncounted p=1
capture); the charter's "counted at runtime" version can only be SLOWER, so the falsifier fires
with margin to spare under the counted requirement too.

CONCLUSION (DIS-candidate): **the T4 low-T lever is the GATED ISA-L engine swap (the +3 prefix
chunks 14->17 that seed-all enabled), NOT faithful runtime placement.** Holding the engine at the
production routing (isal_chunks<=14) and granting rg-grade boundary placement does NOT improve the
T4 wall — it slightly regresses (+34ms: boundary-seed overhead + 1 ISA-L→pure-Rust fallback, with
NO compensating engine win because the marker prefix still pays full u16 cost regardless of where it
lands). This entangles the bootstrap term with the user-GATED engine exactly as the advisor warned;
the faithfully-reachable non-engine placement slice is ~0 (TIE-or-worse) at T4. Mechanism for why rg
is faster here is therefore the ENGINE (rg's clean tail vs gzippy's u16 marker compute on the
window-absent prefix), i.e. LEV-4's 2.3x clean-rate generalized to the prefix — the inner-loop asm
question — NOT a schedulable placement gap.

OWES: synchronous Opus disproof + Steward bankability sign-off (Agent tool absent in owner env =>
self-disproof + pre-registered falsifier only). Harness: new `--kind placement-marker` added to
scripts/bench/{oracle.sh,_oracle_guest.sh} in the worktree (byte-transparent measurement instrument,
not production code) — flag to Steward for review/merge.

## VAR_VIII FULL-KERNEL INLINE-ASM ISOLATION BENCH — DIS-1 REFINED (NEITHER PASS NOR CLEAN-KILL) [2026-06-09, owner/bench-var8-fullkernel, BENCH-ONLY]

> **LEDGER ANNOTATION 2026-06-13** | SCOPE: 2026-06-09, owner/bench-var8-fullkernel, ancestor commit 7bf26096 (d56cb0f5 ancestor; bench-relevant primitives byte-identical), neurotic guest 10.30.0.199 (no_turbo=1), 5 silesia chunks, N=11 interleaved | STATUS: **SCOPED** | CONFIDENCE: 5-chunk bench, sha=OK (byte-exact), rate measurement (moderate); the 0.667x ISA-L ceiling is a bench-only figure for THIS variant and cannot be directly transferred to a fully-restructured production engine. GATE VERDICT (ratio>=0.85 FAIL at 0.667x) still holds as a reference point for assessing future asm efforts; the +14.6% over VAR_VI finding stands. The "DIS-1 PLATEAU spirit upheld via a different mechanism" framing is attribution-level inference (moderate).

Built `VAR_VIII_fullkernel` in `benches/engine_isolation.rs` (BENCH ONLY — NO src/
production touched) — the construct DIS-1/phase2-pass1 OWED: a SINGLE `core::arch::asm!`
hot loop with the back-edge (`jmp 2b`) INSIDE the asm, bit-state(read_in/ril/next_in_pos)
+ output cursor + lit/len short-table base PINNED in registers across it. F2 (speculative
DIST gather, ISA-L small u16 LUT) + D (AVX2 32B MOVDQU non-overlap copy, byte fwd-copy for
overlap/RLE) run IN asm; Rust entered ONLY at long lit/len, long DIST, EOB, or boundary.
12 reg-operands (cold invariants in a KernCtx struct via [ctx+off]); bit engine mirrors
`Bits` exactly (consume_first_decode.rs:245-300).

RUN: frozen guest 10.30.0.199 (bench-lock acquire=quiet runnable_avg=1.00, no_turbo=1,
gov=performance, neighbors frozen; released clean after), taskset -c 0, N=11 interleaved,
seed = real p=1 capture (silesia), 5 swept clean chunks. Built at github origin/reimplement-
isa-l tip **7bf26096** (d56cb0f5 is LOCAL-ONLY/unpushed; 7bf26096 is its ANCESTOR and every
bench-relevant primitive is byte-identical — Bits UNCHANGED, lut_huffman differs only by an
unused heap_bytes() counter, marker_inflate's delta is decode_clean_into_contig which this
standalone bench never calls).

BYTE-EXACT (the absolute gate): **SHA_ALL_EQUAL=yes on ALL 5 chunks** vs VAR_I scalar AND
VAR_III ISA-L FFI oracle. SELFTEST=PASS (iii/i=2.79 in [2.5,3.6]). The AVX copy is byte-correct.

asm_frac (GZIPPY_VIII_COVERAGE): **0.929 / 0.938 / 0.938 / 0.985 / 0.999** (median 0.938);
reentries **3.4K-4.3K** vs VAR_VII's **372K-446K** (~100x collapse). => the DIS-1 per-symbol
asm<->Rust spill confound (asm carried ~1-30%, rate FELL with coverage) is **DECISIVELY
REFUTED** — the full kernel carries 93-99.9% of bytes. The 3 sub-0.97 chunks are long-DIST-
code-heavy (long dist => Rust by design), NOT a careful-tail measurement artifact.

RATE (med-of-med MB/s, vs_iii = /ISA-L; D = AVX, matched to VAR_VI's copy):
  VAR_III ISA-L = 282 (1.000)
  VAR_VIII      = 188 (**0.667**; per-chunk 0.663/0.663/0.696/0.687/0.724)
  VAR_VI(LLVM)  = 164 (0.582; per-chunk 0.581/0.567/0.551/0.556/0.574)
  VAR_VII(per-sym asm) = 81 (0.29)
VAR_VIII materially EXCEEDS VAR_VI by **+14.6% aggregate** (per-chunk +14.6/+16.9/+26.0/
+23.6/+25.7%, SIGN-STABLE on all 5, beyond the 3-9% spread).

PRE-REGISTERED GATE VERDICT (NEITHER clean PASS nor defined KILL — a THIRD outcome):
- byte-exact: PASS. asm_frac>=0.97: median 0.938 (2/5 chunks; DIS-1-confound PURPOSE met
  overwhelmingly, literal 0.97 missed only on long-dist chunks). ratio>=0.85: **FAIL (0.667,
  max 0.724)**.
- KILL condition (VAR_VIII ~= VAR_VI, "register pinning can't beat LLVM"): **REFUTED** — the
  full kernel beats LLVM's best by +14.6% (sign-stable). DIS-1's implied "asm can't beat LLVM"
  PLATEAU does NOT hold for the full kernel; back-edge register-pinning + F2-in-asm IS a real
  lever.
- BUT the full kernel reaches only **0.667x ISA-L** — it closes ~**20%** of the VAR_VI->ISA-L
  gap [(0.667-0.582)/(1.0-0.582)]. The bulk of ISA-L's ~1.5x edge SURVIVES F2+D(AVX)+F4. No
  chunk clears 0.85.

CONSEQUENCE for the user-fork / supervisor's Opus gate: production integration is NOT
authorized (the asm-feasibility-verdict bar: integrate ONLY if VAR_VIII/ISA-L >= 0.85). The
pure-Rust+inline-asm engine does NOT reach ISA-L-class rate even with the register-pinned full
kernel. Per the necessary-NOT-sufficient ceiling (LEV-1/BAR-2): even a PERFECT engine (= ISA-L
= gzippy-isal) loses T1 0.899x / T4 0.900x, so a 0.667x engine cannot clear BAR-1. The route
is bench-bounded: register-pinning helped (~+15%) but the residual 0.67->1.0 gap is intrinsic
to ISA-L's hand-scheduling we did not reach in-process (DIS-1 PLATEAU spirit upheld via a
different mechanism than "== VAR_VI"). OWES: synchronous Opus disproof + Steward sign-off
(Agent tool absent). Bench code on branch bench/var8-fullkernel (uncommitted worktree).

## STRUCTURAL-RESIDUAL SIZING — the owed low-T removal-oracle (var8-gate claim 3) [2026-06-09, owner/structural-residual-sizing @ d56cb0f5, bin 2d317027]

> **LEDGER ANNOTATION 2026-06-13** | SCOPE: 2026-06-09, owner/structural-residual-sizing @ d56cb0f5, bin 2d317027 (== BAR-2 banked isal binary), neurotic guest 10.30.0.199 (no_turbo=1), gzippy-isal, silesia, T1/T4 | STATUS: **SCOPED** | CONFIDENCE: dual-sided removal oracle for serial-output (strong — "PROVED SHARED FLOOR" language is appropriate for a dual-sided removal oracle within this corpus/code/machine scope); causal perturbation + freq-neutral sleep control for marker-bootstrap (strong for criticality, moderate for magnitude — the slow-injection ceiling EXCEEDS the whole T4 gap, confirming rg pays comparable marker work). IMPORTANT: the T1 analysis here ("T1 0.899x is the ENGINE symbol-rate + per-chunk FFI-handoff + serial output floor") was valid at d56cb0f5 but is AFFECTED by the DIS-15 SUPERSEDED finding — the T1 deficit mechanism has changed at HEAD (see DIS-15 annotation). The T4 analysis (marker-bootstrap on critical path, shared with rg, ~0% NET excess) is not known to be superseded.

STEP-0 PROOF-OF-BINARY: freshly built gzippy-isal at HEAD (feature gzippy-isal, x86_64,
RUSTFLAGS=target-cpu=native; built from the main checkout because the fresh worktree's
vendor/isal-rs+zlib-ng SUBMODULES are unpopulated — the 3 uncommitted main src diffs are
COMMENT-ONLY/byte-transparent, the build.rs comment-fix STATE.md §7 wants, codegen-identical
to HEAD). bin_sha=2d3170277a7706e3 (== the BAR-2 banked isal binary, reproducible). env-unset
GZIPPY_VERBOSE: **isal_chunks=14 fb=0 @T4, isal_chunks=16 fb=1 @T1, path=ParallelSM** —
real-ISA-L fingerprint (native stub :390-400 can never increment :386). Frozen guest 10.30.0.199
(bench-lock acquire→quiet runnable_avg 1.25-2.0 <=2.0, no_turbo=1, gov=performance, neighbors
frozen; released clean every run, box thawed). Interleaved N=13-15, same-sink /dev/shm regular
file, sha-verified on byte-exact contenders, vs rg 0.16.0. Instrument = `scripts/bench/{residual.sh,
_residual_guest.sh}` (NEW worktree tooling: multi-contender interleave reusing the parity/oracle
contamination bar verbatim — GZIPPY_* allowlist scrub, host-freeze HARD-FAIL, procs_running
quiet-gate, content-fingerprint stale-binary guard, ParallelSM assert; drives the EXISTING in-src
removal oracle GZIPPY_SKIP_WRITEV_SYSCALL [chunk_fetcher.rs:3911] + the EXISTING marker
slow-knob GZIPPY_SLOW_MARKER_MODE [marker_inflate.rs:1505-1517]). Flag to Steward for merge.

PRODUCTION PARITY (gz-off vs rg-file, per interleaved run; gz wall STABLE, rg the cross-run
variance driver): T1 0.878-0.904x (gz 1018ms / rg 893-921ms); T4 0.861-0.906x (gz 545-547ms /
rg 471-494ms). Quietest runs reproduce the banked T1 0.899 / T4 0.900. The LOAD-BEARING numbers
below are all WITHIN-run (jitter-immune), so the rg cross-run drift does not touch them.

(b) SERIAL-OUTPUT — DUAL-SIDED removal oracle (gz GZIPPY_SKIP_WRITEV_SYSCALL=1 vs rg /dev/null,
one interleaved run): T1 gz-output-share=86ms, rg-output-exposure=85ms (excess +1ms); T4
gz=51ms, rg=44ms (excess +7ms, < spread). Removing output from BOTH leaves the ratio UNCHANGED
(T1 0.878→0.867, T4 0.861→0.862). => **SERIAL-OUTPUT is a PROVED SHARED FLOOR; gzippy-specific
excess ~0% of the gap; NOT a lever.** rg pays an equal serial output (vendor writeFunctor
ParallelGzipReader.hpp:521). (Mildly conservative: gz-skipwritev removes the syscall ENTIRELY
while rg-null still write()s to /dev/null, so gz is favorably biased — the true gz excess is even
smaller / negative.)

(a) MARKER-BOOTSTRAP — causal perturbation (GZIPPY_SLOW_MARKER_MODE) + freq-neutral sleep control
(rule 2) + T1 self-test (rule 4): T4 SPIN +50%→+68ms, +100%→+155ms (monotonic, ~proportional);
T4 SLEEP(yields core) +50%→+23ms, +100%→+67ms (monotonic — criticality SURVIVES the turbo-neutral
control). T1 self-test +100%→+16ms spin/+8ms sleep = FLAT => no marker bootstrap at T1 AND the
instrument is VALIDATED (bites only where marker work exists). gzippy's marker-bootstrap own
critical-path compute @T4 ≈ 59-139ms (the T4−T1 inject diff at F=1.0). **BUT that EXCEEDS the whole
T4 gap (~51ms)** => rg pays comparable marker work (CLAUDE.md: rg carries the SAME u16 marker
machinery, 31-34.5% replaced markers); removing gzippy's entirely would OVERSHOOT rg. And the T4
output-removed ratio (0.862) ≈ T1's (0.867) => the marker bootstrap does NOT widen the gz-vs-rg
ratio beyond the T1 structural floor. => **marker-bootstrap is ON the T4 critical path but is a
SHARED term, ~0% NET gzippy-excess; NOT the lever to >=0.99.**

(c) CHUNK-0 / PER-CHUNK / PIPELINE — the residual after (a)+(b): at T1, engine=ISA-L=rg's engine,
output removed, NO marker bootstrap — yet gz-skipwritev 932ms vs rg-null 808ms = **124ms / 0.867x,
~100% of the output-removed low-T gap, and gzippy-SPECIFIC** (identical engine, rg 13% faster; LOWER
bound per the /dev/null asymmetry above). = per-chunk ISA-L FFI handoff (16 calls) + chunk-0
bootstrap + the ParallelSM worker/consumer/ring/window-map/CRC pipeline gzippy routes through at
EVERY T incl. T1, vs rg's leaner path. **This is a CANDIDATE FAITHFUL-PORT LEVER (rg's per-chunk
window-map + consumer pipeline = the existence proof), NOT removal-proved as irreducible.** Owes its
OWN removal oracle (per-chunk-FFI / chunk-0 / ParallelSM-vs-direct isolation).

VERDICT (var8-gate claim 3): **BAR-1 low-T is NOT a single proved floor.** Two of the three feared
terms are now removal-shown to be NON-levers: OUTPUT = proved SHARED floor; MARKER-BOOTSTRAP = on
critical path but SHARED with rg (removing it overshoots). The live gzippy-specific excess collapses
to the PER-CHUNK/CHUNK-0/PIPELINE term (~13% at T1, engine-matched), which is a candidate lever, not
proved irreducible. => **"native pure-Rust >=0.99 every T is UNREACHABLE" remains the LEADING
HYPOTHESIS, NOT yet removal-proved** — the per-chunk/pipeline oracle is the next owed measurement.
(Scope caveat: this decomposition is on gzippy-ISAL, engine-matched. gzippy-NATIVE additionally
carries the 0.667x VAR_VIII engine wall, so native's BAR-1 needs the engine closed TOO.) OWES:
supervisor's Opus gate + Steward bankability (Agent tool absent => self-disproof + pre-registered
falsifiers + freq-neutral controls + T1 instrument self-test only).

## STEP-0 ISA-L LEVEL-2 DISASM — CLOSED: gzippy & rapidgzip link the IDENTICAL AVX2/BMI2 igzip _04 kernel [2026-06-09, owner/t4-contention @ d56cb0f5, bin 378788924ace0381]

> **LEDGER ANNOTATION 2026-06-13** | SCOPE: 2026-06-09, owner/t4-contention @ d56cb0f5, bin 378788924ace0381, neurotic guest 10.30.0.199, binary disassembly + byte-signature match | STATUS: **SCOPED** | CONFIDENCE: source-read + binary byte-signature verification (strong for the structural finding — both tools link the same AVX2/BMI2 igzip _04 nasm kernel). This is a binary-level finding and does not depend on runtime measurements; it remains valid unless gzippy's ISA-L linkage changes.

PROOF-OF-BINARY: gzippy-isal HEAD, env-unset GZIPPY_VERBOSE: isal_chunks=16/14/14 (T1/T4/T8)
fallbacks=1/0/0, path=ParallelSM (parity assert). bin_sha=378788924ace0381 (== DIS-16 banked isal).
Real ISA-L (native stub gzip_chunk.rs:390-400 can never increment :386).

MACHINE-CODE VERDICT (objdump, read-only; scripts/analysis/disasm_proof.sh, NEW worktree tooling):
- gzippy's compiled ISA-L object igzip_decode_block_stateless_04.o (from .../igzip_decode_block_stateless_04.asm,
  nasm) = 327 insns, **5 VEX (vmovdqu) + 17 BMI2 (shrx/bzhi), 0 legacy SSE** = the AVX2/BMI2 kernel.
  _01.o = 343 insns, 0 AVX/BMI2, 5 SSE (movdqu) = the SSE variant. Both members of libisal.a (ar t).
- LINKAGE (binary is STRIPPED, so symbol-slice impossible — used a contiguous byte-signature): a 24-byte
  contiguous run from _04.o .text (offsets 200/600/1000) is found **byte-VERBATIM in BOTH the stripped
  gzippy binary AND rapidgzip.cpython-313-x86_64-linux-gnu.so** (offset 100 differs in both = reloc region).
- rapidgzip .so is NOT stripped of these statics: exposes decode_huffman_code_block_stateless_{04,01,base,
  dispatch_init,dispatched,mbinit} (the full ISA-L multibinary). gzippy links the SAME _04/_01 objects.
- /usr/local/bin/rapidgzip is a Python wrapper; the real ELF is the .cpython .so (note for future disasm).
=> CONFIRMED: gzippy and rapidgzip execute the SAME AVX2/BMI2 ISA-L _04 inflate kernel, byte-for-byte.
   AVX2/BMI2 instr > 0. ISA-L equivalence Level-2 empirical CLOSED (was the owed "is gzippy's linked
   ISA-L the AVX2 nasm kernel" check). Corroborates DIS-14's source-only "rg statically links _01/_04".

## DIS-17 — "gzippy's T4/T8 gap is memory/cache CONTENTION nondeterminism (17-36% gz variance vs rg 8-10%)" [the charter's NEW-SIGNAL motivating hypothesis] => REFUTED [2026-06-09, owner/t4-contention @ d56cb0f5, bin 378788924ace0381]

> **LEDGER ANNOTATION 2026-06-13** | SCOPE: 2026-06-09, owner/t4-contention @ d56cb0f5, bin 378788924ace0381, neurotic guest 10.30.0.199 (no_turbo=1), gzippy-isal vs rg 0.16.0, silesia, T4/T8, perf stat N=6 | STATUS: **SCOPED** | CONFIDENCE: perf counter measurement (moderate for instruction-count and cache-miss attribution; strong for the false-sharing falsifier — HITM=noise; strong for the "variance disparity does NOT reproduce frozen" finding). ATTRIBUTION NOTE: the "+40% instruction count" finding is per-symbol attribution (perf stat global), NOT a causal perturbation — it is directional evidence consistent with DIS-18 (per-symbol), not itself a verdict. Do not cite as "PROVED cause"; cite as "located excess, confirmed by complementary per-symbol data in DIS-18."

**REFUTED on the bench-locked frozen guest by perf stat (cpu_core P-core PMU, REPS=6) + perf c2c +
peak-RSS, gzippy-isal vs rg 0.16.0. The contention premise does not survive; the gap is WORK
(instruction count) + a small TLB-FOOTPRINT term, NOT contention.** Frozen guest 10.30.0.199 (bench-lock
acquire, no_turbo=1 gov=performance ~1.394 GHz confirmed in counters, released clean), mask 0,2,4,6 (T4) /
0,2,4,6,8,10,12,14 (T8) = P-cores only no SMT-sibling. Output sha-verified byte-perfect by the interleaved
parity harness (perf -r reuses one fd => its own sink-sha mismatch is a -r artifact, NOT a correctness
fail). Drivers scripts/bench/{perf_contention.sh,_perf_contention_guest.sh} (NEW worktree tooling).

WALL+VARIANCE (parity N=15, sha=OK): T4 gz 548ms (spread 4%) / rg 492ms (spread 4%) = **0.898x**;
T8 gz 368ms (spread 10%) / rg 363ms (spread 9%) = **0.985x TIE**. gz and rg run-to-run spreads are
**MATCHED at BOTH cells** (4%/4%, 10%/9%; perf -r elapsed spreads gz 0.9-2.3%). => the charter's
"17-36% gz vs 8-10% rg" variance DISPARITY **does NOT reproduce frozen** — it was thaw/noisy-neighbor
contamination (the DIS-4/INSTR-6 class), NOT a gzippy contention property. The motivating signal is dead.

PERF COUNTERS (mean of 6, T4 / T8):
- **instructions: gz 7.28e9 / 7.38e9 vs rg 5.18e9 / 5.23e9 => gz executes +40% MORE instructions (both T).**
  IPC gz 2.71 vs rg 2.37 (gz's higher IPC partially compensates; wall ratio 0.90 beats the insn ratio 0.71).
- LLC-load-miss MPKI: gz 0.25/0.27 vs rg 0.43/0.44 => gz **0.59-0.62x (LOWER/better)**.
- L1-dcache-miss MPKI: gz 3.05/3.19 vs rg 4.24/4.19 => gz **0.72-0.76x (LOWER/better)**.
- total cache-miss MPKI: gz 2.37/2.54 vs rg 2.43/2.48 => **~EQUAL**.
- dTLB-load-miss MPKI: gz 0.091/0.095 vs rg 0.042/0.050 => gz **1.9-2.2x HIGHER (the ONLY worse cache axis)**.
- page-faults: gz 103,868 / 113,153 vs rg 39,771 / 55,727 => gz **2.0-2.6x more**.
- context-switches: gz 1219 / 764 vs rg 533 / 330 => gz 2.3x more, but ABSOLUTELY tiny (~2200/s; full
  cost <6ms) => lock/sync contention bounded to near-zero as a wall term.
- peak RSS: gz 208 / 295 MiB vs rg 172 / 237 MiB => gz **+21-25% footprint** (~5.4 vs 4.0 MiB/added-thread).

FALSE-SHARING falsifier (perf c2c, gzippy T8; pre-registered "if false sharing: HITM on a named line"):
**Load Local HITM = 6, Remote HITM = 0, Total Shared Cache Lines = 4** over a FULL T8 decode = NOISE FLOOR.
No cache-line ping-pong. Single-socket (no remote). => false-sharing FIRES NEGATIVE.

VERDICT (the located source, with evidence):
1. NO contention source exists: false-sharing HITM=noise; lock ctx-sw tiny; **LLC & L1 MPKI are
   EQUAL-OR-BETTER per instruction** (gzippy is NOT spilling cache more — it is NOT cache-contended).
2. The DOMINANT gap is the **+40% INSTRUCTION COUNT** (gzippy's u16-marker engine + per-chunk ParallelSM
   pipeline do MORE WORK). This is on the critical path by construction and RE-CONFIRMS the banked
   engine-work story (LEV-4 clean-rate 2.3x; VAR_VIII 0.667x asm ceiling) from a fresh perf-counter angle.
   It is WORK, NOT contention.
3. The one located MEMORY difference is **TLB/page-fault FOOTPRINT** (RSS +25%, page-faults 2x, dTLB MPKI
   ~2x) — squarely on the user's "small-shared-hot-in-cache" north star — but a SMALL wall term: the dTLB
   excess (~447k/cell × ~30-cycle page-walk / 1.394 GHz ≈ **9-10ms vs the 56ms/72ms T4/T8 gaps**), and
   DIS-14 already sized the reserve/page-fault axis as wall-SLACK (factor 8 is the tightest 0-fallback
   optimum; faults add only +7-18ms wall at T4 and are identical at T1). The +25% RSS is mostly the D1
   8x output over-reserve + D7 recycle-defer 2-deep hold + per-chunk pipeline buffers.

STEP-2 NOT ENTERED (charter-scoped): STEP-2 is gated on "a specific contention source located" — none was
(every contention falsifier fired negative). The footprint axis IS located + on-goal, but (a) DIS-14
already removal-showed it wall-slack and not freely reducible, (b) the faithful convergences (D1 incremental
output growth; D7 immediate recycle after fixing the CRC-ordering race) are "their own gated turns" per
pipeline-fidelity-verdict.md, and (c) entering it now would re-litigate DIS-14 without a NEW mechanism
(rule 7). OWED FOOTPRINT FALSIFIER (pre-registered for a future supervised turn): "if D1 incremental-growth
+ D7 immediate-recycle cut peak RSS toward rg's (~172/237 MiB) AND cut dTLB MPKI toward rg's, does the T4
wall move >spread? Predicted NULL by DIS-14 (≤~10ms dTLB ceiling, faults slack) — would need to BEAT that
prediction to authorize." gzippy-NATIVE shares this pipeline (+ its 0.667x engine floor); the footprint
finding applies to both, the instruction-count finding is engine-build-specific.

OWES: supervisor's Opus disproof gate + Steward bankability (Agent tool absent => self-disproof + pre-
registered falsifiers + frozen + matched-sink + sha-verified only). NEW worktree tooling to flag to Steward:
scripts/analysis/disasm_proof.sh, scripts/bench/{perf_contention.sh,_perf_contention_guest.sh}.

## DIS-18 — "the +40% pipeline instructions are the per-chunk WRAPPING/SCAFFOLD (resolve/replace 2-pass, u16->u8 narrow, fold/drain, per-chunk CRC, window-map publish, per-chunk init+set_dict, consumer handoff)" [the LOCATE-+40% charter's candidate list] => REFUTED. LOCATED: the +40% IS the pure-Rust u16-MARKER DECODE ENGINE (~57% of user instructions). [2026-06-09, owner/locate-plus40 @ d56cb0f5, symboled gzippy-isal BuildID 6155ce8e]

> **LEDGER ANNOTATION 2026-06-13** | SCOPE: 2026-06-09, owner/locate-plus40 @ d56cb0f5, symboled gzippy-isal BuildID 6155ce8e, neurotic guest 10.30.0.199 (no_turbo=1), silesia, T4, perf record -e instructions N=10472 samples | STATUS: **SCOPED** | CONFIDENCE: **per-symbol perf attribution (weak-to-moderate)** — sample-based perf record gives directional evidence; the ±2-3% sampling spread is explicitly noted; the "~57% marker engine" figure carries that uncertainty. The structural finding ("the suspected wrapping/scaffold is NOT the +40%") is supported by near-zero shares on the suspected items (u16->u8 narrow <0.01%, per-chunk CRC 0.78%) — that null result is more robust than a positive attribution. ATTRIBUTION CAVEAT: per CLAUDE.md rule 1, per-symbol attribution is NOT a verdict; it is a HYPOTHESIS GENERATOR that locates candidates. The headline ("LOCATED: marker engine is the excess") was later corroborated by DIS-19 (rg apples-to-apples split) and DIS-27 (per-core-type instruction count) — those together raise the confidence to moderate. NOTE: there are TWO DIS-18 entries in this file (see convention header); this is the first (per-symbol attribution).

**perf record -e instructions per-SYMBOL attribution on a SYMBOLED gzippy-isal (production codegen: fat
LTO cgu=1 opt3, RUSTFLAGS append `-C strip=none -C debuginfo=2`, instruction count UNCHANGED, 5733 syms,
ELF debug_info not stripped) DECISIVELY locates the excess in the pure-Rust MARKER engine, NOT the
suspected wrapping/scaffold.** Frozen guest 10.30.0.199 (bench-lock acquire/release clean, no_turbo=1
gov=performance ~1.394GHz), mask 0,2,4,6 (T4, P-cores), cpu_core PMU. Output sha-verified byte-perfect
(REFSHA 028bd0...). Drivers scripts/bench/{perf_attr.sh,_perf_attr_guest.sh} (NEW worktree tooling).
Proof-of-binary: path=ParallelSM, isal_chunks=14, isal_fallbacks=0, flip_to_clean=12, finished_no_flip=4,
window_seeded=2, finish_decode=14, inflate_wrapper=0 (==production gzippy-isal at HEAD).

PERF STAT (cpu_core/instructions/u, T4, REPS=6, frozen, sha=OK):
- **gz 6.252e9 instr:u (+-0.99%) vs rg 4.708e9 (+-0.00%) => gz +32.7% USER instructions (~1.54e9 excess).**
  (DIS-17's +40% was user+KERNEL 7.28/5.18=1.405; the +32.7% is the user half, the remainder is the
  kernel/page-fault FOOTPRINT term DIS-17 already located. The symboled build REPRODUCES the excess.)
- cycles gz 2.550e9 vs rg 2.157e9 (+18.2%); IPC gz 2.45 vs rg 2.18 (gz higher IPC partly compensates).
- wall gz 0.531s vs rg 0.470s = **gz 0.885x** (matches the banked T4 ~0.90 scorecard).

PER-SYMBOL SELF instruction share (% of gz 6.25e9 user instr; 10472 samples; DSO 99.63% gzippy binary):
- **MARKER ENGINE (the gzippy-only excess) ~= 56.9% ~= 3.56e9 instr:**
  Block::read_internal_compressed 19.57 + emit_backref_ring 15.89 + SegmentedU16::push_slice 11.29 +
  resolve_chunk_markers_on_chunk 3.52 + HuffmanCodingShortBitsCached::decode 2.91 +
  decode_chunk_unified_marker 2.46 + LutLitLenCode::rebuild_from 0.80 + read_header 0.17 + (init/resolve_range) 0.30.
  Inline tree: decode_chunk_unified_marker -> marker_decode_step -> marker_decode_step_vendor_block ->
  Block::read -> read_internal_compressed -> {LutLitLenCode::decode, Bits::refill/consume} + emit_backref_ring
  -> SegmentedU16::push_slice.
- ISA-L DECODE kernel (SHARED w/ rg, both pay): ~..@37.end 9.82 + ..@42.end 9.13 + loop_block 4.61 +
  decode_len_dist 3.00 + large_byte_copy 1.64 + (decode_literal/multi_symbol_start/..@59/..@52) ~28.3%;
  + ISA-L per-chunk table-setup (make_inflate_huff_code_lit_len 1.08 + setup_dynamic_header 0.45 + ...) ~2.4%.
- gzippy SCAFFOLD ~9.5%: finalize_with_deflate 7.91 + block_finder 1.16 + SegmentedU8::extend 0.25 +
  finalize_window 0.11 + queue_prefetched_marker_postprocess 0.07.
- per-chunk CRC: crc32fast::Hasher::update **0.78%**.

REFUTED SUSPECTS (the charter's candidate list — none is the driver):
- u16->u8 NARROW: does NOT appear (<0.01%) — `resolve_and_narrow_in_place`/`replace_markers_lut_narrow`
  already fused it to ONE in-place pass (byte-exact to vendor DecodedData.hpp applyWindow). Converged.
- resolve/replace "2-pass": resolve_chunk_markers_on_chunk only 3.52% (it is part of the marker engine,
  not a separate 2-pass tax).
- per-chunk CRC 0.78%; window-map publish/queue_prefetched <0.1%; consumer top-of-loop scans (D2/D3)
  <0.1%; per-chunk ISA-L init/set_dict ~2% (shared w/ rg). All SMALL — the suspected wrapping is NOT the +40%.

FALSIFIER (pre-registered: located fns must sum to ~the excess) => PASSES: gz ISA-L ~1.88e9 ~= rg's ISA-L
(same nasm kernel, same clean bytes), so the +1.54e9 USER excess (and ~2.1e9 total incl the kernel/page-fault
footprint) maps to the MARKER ENGINE: gzippy decodes the PREFIX of 12/14 chunks through its u16-marker
inflate (window-ABSENT at speculative decode start, then flip_to_clean) where rapidgzip decodes window-PRESENT
chunks u8-DIRECT via ISA-L (markers only for the small genuinely-window-absent bootstrap; rg's "31.25%
replaced-marker symbols" is over a SMALL chunk fraction). The engine MODEL is FAITHFUL (m_window16 ring +
set_initial_window, deflate.hpp port — in-window backrefs resolve to bytes); the divergence is the FRACTION
decoded via markers, driven by speculation.

CONVERGENCE TARGETS (ranked, for the supervised gate — STEP-2 NOT ATTEMPTED: the clear reduction is
ARCHITECTURAL/scheduler, NOT a byte-transparent one-liner, so per charter I report rather than churn):
1. (owns the ~1.5e9 excess; deepest) Shrink the marker-decoded FRACTION: decode window-PRESENT chunks
   u8-direct via ISA-L from the start (vendor GzipChunk.hpp decodeChunkWithInflateWrapper) instead of
   speculatively u16-marker-decoding their prefix + flipping. == the governing-memory "faithful u8 native
   clean path was never built" + MEMORY confirmed-offset-prefetch-gap (get the predecessor window to the
   worker BEFORE it decodes so flip_to_clean drops from 12/14 toward rg's bootstrap-only).
2. (smaller, residual) The marker engine's own per-symbol efficiency (read_internal_compressed +
   emit_backref_ring + push_slice) — only worth it for the genuinely-window-absent fraction rg also pays.

OWES: supervisor's Opus disproof gate. NEW worktree tooling to flag to Steward: scripts/bench/{perf_attr.sh,
_perf_attr_guest.sh}. NOTE: the guest /root/gzippy-bench/target/release/gzippy is now the SYMBOLED
gzippy-isal binary (overwrote the staged native bench binary) — next parity turn's --build rebuilds it; a
--no-sync reuse expecting native must rebuild first.

## DIS-19 — "rapidgzip's marker-decode instruction cost ~= 0 (the DIS-18 FALSIFIER's hidden premise; rg ISA-L-u8-directs window-present chunks, markers only a small bootstrap)" => REFUTED. MEASURED: rg marker-decodes ALL chunks; the +1.54e9 gz excess splits ~71% marker INNER-LOOP / ~25% resolution-SCAFFOLD. [2026-06-09, rg-marker-attr worktree, ALREADY-SYMBOLED rapidgzip 0.16.0 .so BuildID 67cd8b7e]

> **LEDGER ANNOTATION 2026-06-13** | SCOPE: 2026-06-09, rg-marker-attr worktree, rapidgzip 0.16.0 .so BuildID 67cd8b7e, neurotic guest 10.30.0.199 (no_turbo=1), silesia, T4, perf record N=8990 samples | STATUS: **SCOPED** | CONFIDENCE: **source-read (strong for structural findings) + per-symbol attribution (weak-to-moderate for percentage splits)**. The source-read finding ("rg marker-decodes EVERY chunk via decodeChunkWithRapidgzip, NEVER calls decodeChunkWithInflateWrapper<IsalInflateWrapper> for streaming -d -c") is strong (zero perf samples on that path, source-confirmed). The percentage split (71% inner-loop / 25% scaffold) carries ±2-3% sampling uncertainty (noted in text). The apples-to-apples EXCESS TABLE is the key artifact; the "~71%/~25%" labels should be read as directional, not precise. ATTRIBUTION CAVEAT: the split identifies sub-levers to prioritize, NOT verified levers; per CLAUDE.md rule 1 each sub-lever still requires a causal perturbation. "DECISIVELY REFUTED" language for the structural find ("rg uses igzip inside its marker engine") is appropriate; for percentage splits it should be read as "approximately located."

**The OWED apples-to-apples counterpart of DIS-18.** DIS-18 perf-recorded gzippy-isal's per-symbol share
and ASSUMED rg-marker ~= 0. The installed rapidgzip 0.16.0 .so is ALREADY symboled (debug_info, not
stripped, 7016 nm syms), so NO build was needed — just perf record on rg. Frozen guest 10.30.0.199
(bench-lock acquire/release CLEAN, no_turbo=1 during, released no_turbo=0), mask 0,2,4,6 (T4, P-cores),
cpu_core PMU, REPS=6, output sha-verified byte-perfect (REFSHA 028bd0...). Drivers
scripts/bench/{rg_marker_attr.sh,_rg_marker_attr_guest.sh} (NEW worktree tooling, flag to Steward).

PERF STAT (cpu_core/instructions/u, T4, REPS=6, frozen, sha=OK):
- **rg 4.7078e9 instr:u (+-0.00%)** — EXACTLY reproduces DIS-18's 4.708e9. cycles 2.187e9, wall 0.4797s.
  (gz was 6.252e9 => +1.54e9 / +32.7% excess, unchanged.)

STRUCTURAL FINDINGS (source + address + perf confirmed):
- **rg NEVER calls decodeChunkWithInflateWrapper<IsalInflateWrapper> (the u8-direct ISA-L path): 0 samples.**
  For streaming -d -c, rg marker-decodes EVERY chunk via decodeChunkWithRapidgzip -> Block<false>::read
  (vendor GzipChunk.hpp:709/719), resolving markers afterward via DecodedData::applyWindow. The
  "rg u8-directs window-present chunks" premise is FALSE for streaming decode (it is only the
  index/random-access path: initialWindow && untilOffsetIsExact). So gzippy's flip_to_clean (12/14 chunks
  flip to REAL ISA-L FFI) is itself a DIVERGENCE from rg, which marker-decodes the whole chunk with an
  ISA-L Huffman PRIMITIVE (HuffmanCodingISAL) and resolves via applyWindow.
- **rg uses the SHARED igzip AVX2 nasm kernel INSIDE its marker engine.** Address-confirmed: ..@37.end
  (0x1967fe), ..@42.end (0x1968ce), ..@59.end all lie within decode_huffman_code_block_stateless_04
  (0x1966d0) — the same kernel gzippy links. Reached via decodeChunkWithRapidgzip (the only decode entry
  called) -> Block::read -> HuffmanCodingISAL.

RG PER-SYMBOL SELF share (8990 samples, period 300000, DSO 92.57% rg .so; % of rg 4.7078e9):
- SHARED igzip kernel ~35.6%: ..@37.end 11.40 + ..@42.end 10.55 + loop_block 4.69 + decode_len_dist 3.36 +
  large_byte_copy 1.56 + make_inflate_huff_code_lit_len 1.76 + set_and_expand 0.62 + setup_dynamic_header
  0.58 + HuffmanCodingISAL::init 0.22 + (set_codes/inflate_in_load/..@59/..@52/small_byte_copy) ~0.35.
- MARKER INNER LOOP ~47.7%: Block<false>::read 41.75 + BitReader::peek2 3.38 + decodeChunkWithRapidgzip
  (self) 1.07 + HuffmanCodingReversedBitsCached::init 0.57 + readHeader 0.17 + setInitialWindow 0.26 + read2 0.13.
- MARKER RESOLUTION (scaffold) ~7.12%: DecodedData::applyWindow 6.65 + getWindowAt 0.47.
- misc shared: __memmove_avx 4.63 + crc32 1.51 + blockfinder seekToNonFinal* 0.84 + memcpy@plt 0.40 + python ~0.7.

APPLES-TO-APPLES ABSOLUTE SPLIT (self-attribution basis, == DIS-18 methodology):
| axis | gzippy (x6.252e9) | rapidgzip (x4.7078e9) | gz EXCESS |
|------|------|------|------|
| shared igzip kernel | ~30.7% = 1.88e9 | ~35.6% = 1.68e9 | ~0 (shared, equal) |
| marker INNER LOOP | 53.4% = 3.338e9 | 47.7% = 2.246e9 | **+1.09e9 (~71%)** |
| marker RESOLUTION/scaffold | 11.4% = 0.715e9 | 7.12% = 0.335e9 | **+0.38e9 (~25%)** |
| (sum non-kernel marker pipeline) | 4.053e9 | 2.581e9 | +1.47e9 ~= the +1.54e9 |
  gz INNER LOOP = read_internal_compressed 19.57 + emit_backref_ring 15.89 + push_slice 11.29 +
  HuffmanShortBitsCached 2.91 + decode_chunk_unified_marker 2.46 + LutLitLenCode 0.80 + header/init 0.47.
  gz RESOLUTION = resolve_chunk_markers 3.52 + finalize_with_deflate 7.91.

FALSIFIER (pre-registered): {rg inner-loop ~= gz & gap is SCAFFOLD => faithful-convergeable} vs
{gz inner-loop heavier per byte => open inner-loop-engine, predict VAR_VIII-style plateau ~0.667x}.
=> **RESOLVES to the INNER-LOOP branch.** gzippy's marker inner loop is +1.09e9 heavier on the SAME ~73M
markers (+49% per-byte) and is the DOMINANT axis (~71%); resolution/scaffold is real but secondary (~25%).
So this is NOT a pure-scaffold faithful-convergeable gap — the lever lands on open-inner-loop territory.

INNER-LOOP EXCESS DECOMPOSED (for the gate — it is not monolithic):
- u16 OUTPUT/BACKREF ABSTRACTION ~1.70e9: push_slice 0.706e9 (SegmentedU16 buffer-append) +
  emit_backref_ring 0.993e9 (u16-RING modular-index copy). rg FUSES the equivalent into Block<false>::read's
  inlined FLAT m_window16[pos++] writes (near-free). This is the largest single tractable sub-lever:
  flatten the segmented/ring buffer + inline the writes + fuse decode->track->output into ONE function like
  rg's Block::read. STRUCTURAL/architectural (per DIS-18 STEP-2-not-attempted), not a one-liner.
- pure-Rust Huffman decode ~1.61e9: read_internal_compressed 1.224e9 + HuffmanShortBitsCached 0.182e9 +
  decode_chunk_unified_marker 0.154e9 + LutLitLenCode 0.050e9. rg uses the ISA-L igzip kernel as its
  Huffman primitive here. Matching it = the inner-loop-asm question — VAR_VIII full register-pinned asm
  plateaued at 0.667x ISA-L, so a similar plateau is PREDICTED.

BAR-1 READ (honest): this COMPLETES the isal attribution. The final isal lever = gzippy's per-byte
u16-MARKER pipeline is ~1.5x rg's on the matched fraction, dominated (71%) by the marker INNER LOOP
(open-inner-loop-engine, plateau-predicted), with a 25% resolution/scaffold tail (faithfully convergeable
but page-warmth already sized as wall-slack). gzippy-NATIVE shares this marker engine PLUS its 0.667x
clean-path floor. => closing isal T4 0.91x -> 0.99 by converging the marker engine alone is UNLIKELY to
clear the bar (VAR_VIII precedent); the largest tractable move is fusing gzippy's fragmented
read_internal_compressed -> emit_backref_ring -> push_slice (3 functions + SegmentedU16/u16-ring) into
rg's single inlined flat-buffer Block::read (~1.70e9 of the excess is this abstraction fragmentation).

OWES: supervisor's Opus disproof gate. CAVEAT for the gate: percentages are single-capture period-sampled
(8990 samples) on the SELF basis to match DIS-18; the absolute split carries ~+-2-3% sampling spread, but
the headline (inner-loop is the dominant axis, scaffold secondary, kernel shared-equal) is robust to it.
Clean: guest find-orphans from exploration KILLED (load decaying), local wrapper killed, bench-lock released.

---

## DIS-20 — "the u16 OUTPUT/BACKREF FRAGMENTATION (emit_backref_ring + marker drain push_slice; DIS-19's ~1.70e9 sub-lever) is WALL-SLACK like the page-warmth footprint axis (TIE-6), so the faithful flat-buffer port is NOT worth funding" => REFUTED at T4 (de-frag region is FIRMLY ON THE T4 CRITICAL PATH); CONFIRMED-slack at T1. [2026-06-09, owner/defrag-wall-oracle worktree @ d56cb0f5, gzippy-isal bin b0a4d0cf fingerprint fa045c9d, frozen guest 10.30.0.199]

> **LEDGER ANNOTATION 2026-06-13** | SCOPE: 2026-06-09, owner/defrag-wall-oracle @ d56cb0f5, bin b0a4d0cf (fingerprint fa045c9d), neurotic guest 10.30.0.199 (no_turbo=1), gzippy-isal, silesia, T4/T1, GZIPPY_SLOW_DEFRAG slow-injection, N=15 | STATUS: **SCOPED** | CONFIDENCE: **causal perturbation + freq-neutral sleep control (strong for criticality)**. Monotonic slope (+243/+431/+772ms at T4) survives sleep control (+177ms) — criticality at T4 is firmly established. T1 slack confirmed (+37ms@200%, small). Per CLAUDE.md rule 3: slow-down slope ≠ speed-up ceiling — the ceiling is NOT bounded by this perturbation; see DIS-21 for the ceiling measurement. NOTE: a hit-counter contention confound was caught and corrected before the final numbers (detailed in text).

**The owed causal perturbation the rg-marker gate (plans/rg-marker-completing-gate-verdict.md CLAIM B FIX-NEEDED) demanded before any de-frag PORT.** Method: byte-transparent SLOW-injection knob `GZIPPY_SLOW_DEFRAG` (slow_knob.rs) injecting busy-ALU work PROPORTIONAL TO u16 BYTES MOVED at the EXACT DIS-19 de-frag sites — inside `emit_backref_ring` (u16-ring modular copy, prop. to length) and the `drain_to_output` marker branch (SegmentedU16 push_slice, prop. to new_bytes). This is the CLAUDE.md rule-1 perturbation TEMPLATE (the cheapest falsifier first; per rule 3 it sizes CRITICALITY, not the speed-up ceiling). NOT the full flat-buffer removal oracle (that is most of the very port we are deciding to fund, + high byte-exactness risk).

INSTRUMENT VALIDATION (rule 4): byte-transparent — OFF sha == ON-spin sha == ON-sleep sha == ref 028bd002… (proved locally on a 24MB fixture AND every guest arm sha=OK). Site fires 137.15M u16/decode on silesia (cross-checks DIS-19's ~1.70e9: 137M × ~10-12 instr/u16 in the segment-append + ring-modular path ≈ 1.6e9). CONFOUND CAUGHT + corrected: run-1 forced GZIPPY_SLOW_HITS=1 whose shared `DEFRAG_HIT_BYTES.fetch_add` CONTENDS across 4 workers at T4 (+140–214ms cache-line-ping-pong artifact). Re-ran CLEAN (hit-counter OFF); numbers below are clean per-thread-local spin.

RAW (frozen guest, bench-locked runnable_avg 1.25–1.75 quiet, interleaved N=15, sha=OK every arm, path=ParallelSM, isal_chunks=14/0@T4 16/1@T1, vs rg 0.16.0; same-sink OFF baseline = production, no knob/atomic):
| arm | T4 wall | Δ vs OFF | ratio_vs_rg | | T1 wall | Δ vs OFF | ratio |
|-----|------|------|------|--|------|------|------|
| OFF (same-sink) | 551ms | — | 0.891 (spread 7%) | | 1026ms | — | 0.898 (spread 1%) |
| DEFRAG=100% spin | 794ms | **+243ms** | 0.625 | | — | — | — |
| DEFRAG=200% spin | 982ms | **+431ms** | 0.504 | | 1063ms | **+37ms** | 0.865 |
| DEFRAG=400% spin | 1323ms | **+772ms** | 0.373 | | — | — | — |
| DEFRAG=200% SLEEP (freq-neutral control) | 728ms | **+177ms** | 0.678 | | — | — | — |

VERDICT (the de-frag-pays-or-wall-slack decision):
- **T4: de-frag region is FIRMLY ON THE CRITICAL PATH.** Monotonic +243/+431/+772ms (6–19× the ~40ms spread); the SLEEP control still adds +177ms ⇒ criticality SURVIVES the frequency-neutral swap (NOT a busy-spin turbo artifact, rule 2). This **REFUTES the wall-slack null branch of the pre-registered falsifier AT T4.** TIE-6's "u16 footprint axis = wall-slack" prior does NOT extend to the de-frag DATA-MOVEMENT region — page-warmth/footprint (which TIE-6 sized) and on-critical-path data-movement are DIFFERENT axes.
- **T1: de-frag region is NEARLY SLACK** (+37ms@200%, ~3.7× the 1% spread but small ~4% of wall). T1 wall is SERIALIZATION-bound (DIS-15: each chunk waits the prior's 32 KiB window), behind which the single worker's de-frag overlaps with handoff stalls. So de-frag is NOT the T1 lever (T1 is the routing/single-shot problem per DIS-15).
- **Mechanistically coherent:** at T4 the 4 workers ARE the binder (consumer waits on worker output, DIS-16) and de-frag runs ON the workers + gates each chunk's marker-prefix→flip-to-clean, so slowing it raises the wall ≥1:1; at T1 the binder is serial handoff, not worker compute.

OWES / NEXT (rule 3 — slow-slope ≠ speed-up ceiling): this SLOW-injection establishes CRITICALITY, NOT the speed-up payoff. The fraction of the ~60ms T4 gap a FAITHFUL FLAT-BUFFER port (decode straight into one flat aligned m_window16 w/ inlined window[pos++] + single memcpy back-ref, no ring/drain/segment-list — deflate.hpp:926/1319/1376) recovers is NOT bounded here. The de-frag flat-buffer port is now **WARRANTED at T4** (not refuted as slack; the largest faithful sub-lever per DIS-19) — the OWED next step is the flat-buffer REMOVAL oracle (or the direct faithful port + interleaved sha-verified measure vs rg) to size the recovered fraction. Do NOT prejudge the magnitude; do NOT touch flip-to-clean (CONVERGENT, rg-marker gate).

PROVENANCE: branch owner/defrag-wall-oracle; slow_knob.rs `GZIPPY_SLOW_DEFRAG`/`inject_defrag`; marker_inflate.rs `emit_backref_ring`+`drain_to_output` inject sites; scripts/bench/{oracle.sh `--slow-kind`, _oracle_guest.sh perturb GZ_ENV magnitude-on-named-knob + SLOW_HITS opt-in, defrag_drive.sh, defrag_drive2.sh}. Logs /tmp/defrag-drive{,2}. NEW tooling — FLAG TO STEWARD. Clean: local timeout-wrapper orphans killed, guest no stray gzippy/find, neurotic no_turbo=0 frozen_now=[] watchdog=inactive (freeze released). OWES: supervisor's Opus disproof gate.

## DIS-21 — "the FAITHFUL FLAT-BUFFER de-frag port (DIS-20's WARRANTED next step) recovers a MEANINGFUL fraction of the ~70ms T4 gap" => REFUTED on TWO independent grounds: (1) SOURCE — there is NO flat per-chunk buffer in rapidgzip to converge toward; gzippy is ALREADY structurally convergent; (2) MEASURED — the only faithful byte-exact convergence available (emit_backref_ring's copy -> rg's EXACT single memcpy) is a clean TIE at T4. The de-frag flat lever is a PHANTOM born from misreading `m_window16`. [2026-06-09, owner/defrag-flat-convergence worktree @ d56cb0f5, gzippy-isal bin 9b15bc89, frozen guest 10.30.0.199, isal_chunks=14/0]

> **LEDGER ANNOTATION 2026-06-13** | SCOPE: 2026-06-09, owner/defrag-flat-convergence @ d56cb0f5, bin 9b15bc89, neurotic guest 10.30.0.199 (no_turbo=1), gzippy-isal, silesia, T4 | STATUS: **SCOPED** | CONFIDENCE: **source-read (strong for the structural finding that rg HAS the ring/drain/segment-list) + removal-oracle-grade for the copy-mechanism TIE (byte-exact interleaved OFF/ON, N=15)**. The source refutation ground is strong: deflate.hpp:805/1319/1376/1369-1390 establish that rg's m_window16 is a ring, not a flat buffer — first-hand re-derivation cited in text. The measured TIE ground is strong: -3ms (min) / -1ms (med), within spread, with the instr-count confirmation that the copy-win does NOT translate to wall (off-critical-path). Do NOT re-attempt the "flat per-chunk buffer" direction without a NEW mechanism that differs from the cited vendor source.

**(1) SOURCE REFUTATION (first-hand vendor re-derivation — the rg-marker gate's owed `m_window16` read).** DIS-19/DIS-20/the defrag-wall-oracle charter all rest on "rg decodes straight into ONE flat aligned `m_window16` w/ inlined `window[pos++]` + single memcpy, NO ring/drain/segment-list (deflate.hpp:926/1319/1376)." That is a MISREAD. Re-derived first-hand:
- `deflate.hpp:805` `using PreDecodedBuffer = std::array<uint16_t, 2 * MAX_WINDOW_SIZE>` — `m_window16` is a 65536-u16 RING, NOT a per-chunk-sized flat buffer. It is BYTE-FOR-BYTE gzippy's `output_ring` (RING_SIZE = 2*MAX_WINDOW_SIZE). Vendor comment :796-801 confirms: "circular buffer ... round up to power of two ... modulo can be a bitwise 'and'." Same modulo-by-AND gzippy uses.
- `resolveBackreference` (:1369-1390) ≡ gzippy `emit_backref_ring`: SAME `offset = (m_windowPosition + size - distance) % size`, SAME single-`memcpy(&w[pos], &w[offset], length*2)` no-wrap fast path (:1376), SAME backward marker scan (:1379-1389). `appendToWindow` (:1319-1322) is `window[m_windowPosition++] = sym; pos %= size` — the SAME `% RING_SIZE` ring write, NOT a flat append.
- `read()` (:1288) `result.dataWithMarkers = lastBuffers(m_window16, m_windowPosition, nBytesRead)` then `DecodedData::append` (DecodedData.hpp:266-275) `appendToEquallySizedChunks(dataWithMarkers, buffer)` into 128 KiB / 64Ki-u16 SEGMENTS — BYTE-FOR-BYTE gzippy's `drain_to_output` -> `SegmentedU16::push_slice` (SEGMENT_ELEMENTS=64Ki). rg HAS the ring, HAS the per-read drain, HAS the segment-list. There is nothing "flat" to converge to.
=> gzippy's u16 marker output/backref path is ALREADY a faithful structural port of rg's. Building a "flat per-chunk buffer (no ring/drain/segment-list)" would DIVERGE from rapidgzip (bias-guardrail violation), not converge. This satisfies CLAUDE.md rule-7(a): the rejection mechanism IS "rg does the de-frag the SAME way." The DIS-19 "shared igzip kernel ≈ equal" leg already hinted this.

**(2) MEASURED RULE-3 REMOVAL BOUND (the one faithful byte-exact convergence that exists).** The sole genuine micro-divergence: gzippy's `emit_backref_ring` non-overlap arm uses a word-copy-rounded-to-4 fast path (a documented gzippy INNOVATION over vendor); rg uses a plain `std::memcpy(dst, src, length*2)` (:1376). Built a byte-exact gated oracle `GZIPPY_FLAT_BACKREF=1` (slow_knob.rs `flat_backref_enabled`, marker_inflate.rs non-overlap arm -> `copy_nonoverlapping(length)`; the ≤3-u16 word-copy overshoot is invisible so OFF-sha==ON-sha). This is the charter's "single-memcpy back-ref" done the ONLY faithful (convergent, not divergent) way, and the BIGGEST DIS-19 sub-piece (emit_backref_ring 0.993e9).

RAW (frozen guest 10.30.0.199, bench-locked runnable_avg 1.00 <=2.0 quiet, INTERLEAVED OFF/ON/rg in ONE loop N=15, byte-exact OFF sha==ON sha==ref 028bd002…, path=ParallelSM, isal_chunks=14 isal_fallbacks=0, vs rg 0.16.0; scripts/bench/flat_ab_interleave.sh):
| arm | T4 min | T4 med | spread | ratio_vs_rg |
|-----|--------|--------|--------|-------------|
| OFF (production word-copy) | 552.4ms | 565.4ms | 5.7% | 0.874 |
| ON (flat-backref = rg memcpy) | 549.7ms | 564.1ms | 5.3% | 0.878 |
| rapidgzip 0.16.0 | 482.6ms | 495.7ms | 6.7% | — |
**ON-OFF delta = -3ms (min) / -1ms (med)** — within the ±30ms (5-6%) spread. CLEAN TIE.
(Cross-check, two SEPARATE harness oracle.sh runs same binary 9b15bc89: same-sink OFF 555ms ratio 0.894 @runnable2.0; flat-backref ON 544ms ratio 0.911 @runnable1.0 — the 11ms there was the OFF arm's higher load, dissolved in the interleaved run.)

MECHANISM CHECK (instr-count, perf stat -e instructions:u T4, 3 reps): OFF ~6.488e9, ON ~6.392e9 => ON cuts ~96M instr (~1.5%). So the flat copy DOES cut instructions yet the wall is FLAT — the TEXTBOOK rule-3 signature (an instruction win that does NOT translate to wall: the back-ref copy is off the critical path / memory-latency-hidden). NOTE: this swaps only the copy MECHANISM (~96M), far less than the full 0.993e9 emit_backref_ring (whose offset-modulo + marker-scan + the drain push_slice are all EXACT vendor counterparts per (1), hence NOT faithfully removable).

VERDICT (the de-frag-pays-or-low-ceiling decision DIS-20 owed): **LOW-CEILING — the de-frag flat-buffer port is NOT worth funding.** DIS-20's criticality is real and stands (slowing the region adds wall, slope confirmed) BUT the SPEED-UP ceiling for converging it to rg's shape is ~0ms, because (a) gzippy is already at rg's structure (ring+drain+segments) and (b) the one faithful copy-mechanism convergence TIEs while cutting 1.5% instr. The isal T4 0.87-0.91 gap is therefore NOT in the u16 output/backref fragmentation — it lies in the OTHER marker half (the ~1.61e9 marker-prefix Huffman SYMBOL RATE, asm-bounded/VAR_VIII-plateau, user-gated) and/or parallel SCHEDULING (MORNING-BRIEF DIS-16/17), NOT a faithful flat-buffer lever. isal T4 is near-floor for the faithful non-asm path on the marker-output axis. The push_slice/drain half (DIS-19 0.706e9) was NOT separately measured but is the SAME conclusion by (1): it is `appendToEquallySizedChunks`, an exact vendor counterpart — nothing faithful to converge.

PRE-REGISTERED FALSIFIER OUTCOME: the falsifier said "ON ~= OFF (wall flat despite instr drop) => next binder binds immediately => de-frag on-critical-path-but-LOW-CEILING, NOT worth the full port." That is EXACTLY what was observed (-3ms wall, -1.5% instr). The de-frag flat port is REFUTED as a lever.

PROVENANCE: branch owner/defrag-flat-convergence @ d56cb0f5; slow_knob.rs `flat_backref_enabled`/`GZIPPY_FLAT_BACKREF`; marker_inflate.rs emit_backref_ring non-overlap arm; scripts/bench/{oracle.sh + _oracle_guest.sh `flat-backref` KIND (byte-exact, CHECK_SHA=1), flat_ab_interleave.sh}. NEW tooling — FLAG TO STEWARD. NOTE: worktree `vendor/` was repopulated from main's submodules (worktree-add leaves submodules empty; rsync --delete had wiped guest isal-sys on the first try — fixed, rebuilt clean). Clean: local timeout-wrapper sleep-sentinel orphans killed (14 reaped), guest+neurotic no stray gzippy/rapidgzip/perf, neurotic no_turbo=0 watchdog=inactive (freeze released). OWES: supervisor's Opus disproof gate.

## DIS-22 — "Routing single-threaded gzippy-isal single-member decode to ONE ISA-L call (IsalSingleShot), instead of forcing the 16-chunk ParallelSM pipeline, IS a production WIN at T1 and is byte-exact, multi-member-safe, and T4/T8-neutral" => CONFIRMED. [2026-06-09, owner/t1-singleshot-route worktree @ d56cb0f5+route, gzippy-isal bin eceafeea1970b340, frozen guest 10.30.0.199 no_turbo=1 runnable_avg=1.00, vs rg 0.16.0, corpus sha 028bd002…cb410f]

> **LEDGER ANNOTATION 2026-06-13** | SCOPE: 2026-06-09, owner/t1-singleshot-route @ d56cb0f5+route, bin eceafeea1970b340, neurotic guest 10.30.0.199 (no_turbo=1), gzippy-isal, silesia, T1/T4/T8 | STATUS: **STALE-RISK** | The routing change (IsalSingleShot for gzippy-isal T1 single-member) IS now in the working tree (confirmed: src/decompress/mod.rs at HEAD contains DecodePath::IsalSingleShot). The **performance WIN basis (1.200x at T1) rests on DIS-15's 247ms ParallelSM overhead**. DIS-15 NOTE: the solvency 6ms measurement that was cited as "DIS-15 SUPERSEDED" was retracted (commit 39277bfb: FORCE_PARALLEL_SM dead no-op at HEAD; the 6ms compared shot-vs-shot, not PSM-vs-shot); DIS-15's 247ms is NOT refuted. So DIS-22's 1.200x figure rests on a finding (DIS-15 247ms) that remains valid in its neurotic/d56cb0f5 context. HOWEVER: the 1.200x was measured at neurotic; the solvency box (AMD Zen2, PEXT microcoded) has a different ISA-L performance profile — re-measure on solvency before citing as a wall win there. The CORRECTNESS and byte-exactness of the route are not known to be affected (887 tests passed; multi-member-safe classification is structural). CONFIDENCE of original finding: removal-oracle-grade for the performance win on neurotic/gzippy-isal/silesia (reproduces DIS-15's prediction to the digit); CONFIDENCE for solvency: unverified (STALE-RISK — different arch).

THE CHANGE (production routing, src/decompress/mod.rs): new `DecodePath::IsalSingleShot`. In `classify_gzip`, under `#[cfg(all(parallel_sm, isal_clean_tail))]` (gzippy-isal x86_64 only), `num_threads <= 1` for a SINGLE-MEMBER stream returns `IsalSingleShot`; the dispatch arm calls `isal_decompress::decompress_gzip_stream` (one `isal_inflate`, gzip CRC32+ISIZE verified via IGZIP_GZIP crc_flag, non-zero ret => terminal Err, NO fallback — rule 5). BGZF + multi-member are classified ABOVE this route (unaffected); gzippy-native (`isal_clean_tail` false) stays ParallelSM at every T. This is the DIS-15 lever turned into production routing.

MEASURED (parity.sh spine, --feature gzippy-isal, frozen-guest, interleaved gzippy vs rg in ONE loop, sha-verified EVERY trial, routing asserted via GZIPPY_DEBUG path=):
| cell | path (asserted) | N | gzippy min | rg min | ratio | verdict | sha |
|------|-----------------|---|-----------|--------|-------|---------|-----|
| T1   | IsalSingleShot  | 15 | 0.7659s (277 MB/s, spread 1%) | 0.9194s (231 MB/s) | **1.200x** | **WIN(gzippy)** | OK |
| T4   | ParallelSM      | 11 | 0.5489s (386 MB/s, spread 3%) | 0.4975s (426 MB/s) | 0.906x | LOSS (pre-existing) | OK |
| T8   | ParallelSM      | 11 | 0.3606s (588 MB/s, spread 31%) | 0.3742s (566 MB/s) | 1.038x | TIE | OK |

T1 reproduces DIS-15's single-shot prediction (1.197x) to the digit. T4 0.906x and T8 1.038x are UNCHANGED vs the prior banked ParallelSM numbers (DIS-15/16: T4 0.905–0.911x) => the T1 route does NOT regress T4/T8 (they stay ParallelSM; the route is gated on num_threads<=1). The T4 LOSS is the pre-existing parallel-scheduling gap (LEV-2/DIS-16), NOT introduced here.

CORRECTNESS: (1) full lib suite gzippy-isal `--test-threads=1` = **887 passed, 0 failed, 11 ignored** (incl. multi-member, routing/deletion-trap `test_single_member_routing_multithread`, silesia CRC stress, trace_parity bgzf/gz_subfield/multi_member, three-oracle differential, and `test_parallel_sm_thread_gate` asserting T1=IsalSingleShot / T2-4=ParallelSM on isal). (2) MULTI-MEMBER-at-T1 (the swallow risk): a `cat a.gz b.gz` stream at T1 routes to **MultiMemberSeq** (NOT IsalSingleShot — classified earlier) and is byte-exact at T1 AND T4. (3) DUAL-SHA both features at T1/T4/T8 = byte-exact (pin 028bd002…); gzippy-NATIVE stays ParallelSM at every T (route cfg-gated off) — byte-transparent.

PROVENANCE: worktree .claude/worktrees/t1-singleshot-route, branch owner/t1-singleshot-route (uncommitted working tree @ d56cb0f5 base); src/decompress/mod.rs (IsalSingleShot variant+route+dispatch+thread-gate test), src/tests/diff_ratio.rs (T1-guard now calls decompress_parallel(...,1) directly so it stays pipeline-vs-itself), scripts/bench/{parity.sh,_parity_guest.sh} (EXPECT_PATH knob for the routing assertion). NOTE: worktree submodules were empty (worktree-add trap, same as DIS-21) — repopulated vendor/isal-rs + vendor/isa-l via `git submodule update --init` before the build. Clean: local timeout-wrapper sleep-sentinel orphans reaped (the ~/.dotfiles/bin/timeout wrapper sleeps out its full duration post-completion — leaked 14 wrappers+ssh children+3 ensure-corpus hops, all killed), guest no stray gzippy/cargo, neurotic no_turbo=0 watchdog=inactive (freeze released). OWES: supervisor's Opus gate.

## DIS-23 — "FAITHFUL INCREMENTAL OUTPUT GROWTH (rg GzipChunk.hpp:309-379 port) cuts gzippy's footprint (RSS + dTLB MPKI) toward rg's" (the DIS-17 owed footprint falsifier, user's cache-locality north star) => SPLIT: dTLB half CLOSED (drops BELOW rg, wall-neutral, 0-fallback, byte-exact); RSS half NOT closed (lazy-faulting + realloc transient). [2026-06-09, owner/isal-incremental-growth worktree @ d56cb0f5, gzippy-isal bin 908a9629eab96667, frozen guest trainer/10.30.0.199]

> **LEDGER ANNOTATION 2026-06-13** | SCOPE: 2026-06-09, owner/isal-incremental-growth @ d56cb0f5, bin 908a9629eab96667, neurotic guest 10.30.0.199 (no_turbo=1), gzippy-isal, silesia, T4/T8; lib suite 893 passed | STATUS: **SCOPED** | CONFIDENCE: **strong for the dTLB MPKI win (perf stat N=6, N=3 independent runs, wall-neutral, 0-fallback, byte-exact); strong for the RSS null result (mechanism explained: lazy-faulting means peak RSS tracks touched bytes, not reserved capacity; f1/f2 raise RSS via realloc churn)**. The DIS-14 source-claim REFUTATION (rg uses 128KiB incremental growth, not the whole buffer) is strong (source-read GzipChunk.hpp:309-379). NOTE: the gated change (GZIPPY_ISAL_INCREMENTAL_GROWTH) was recommended for a default-ON flip but with FACTOR=4; DIS-29 later found that always-small factor-4 REGRESSES sub-8x corpora at T1 — the recommended production form is RETRY-ON-None-WITH-GROWTH (keep 8x upfront reserve, grow only on overflow). Cite DIS-29 alongside DIS-23 when discussing the default-ON decision.

**(0) SOURCE — DIS-14's "rg's avail_out feed is the WHOLE buffer (isal.hpp:258), same as gz; 'rg refills in small chunks' is FALSE" is REFUTED first-hand.** `isal.hpp:254-258` `readStream(output, outputSize)` feeds `avail_out = outputSize` — but the CALLER `finishDecodeChunkWithInexactOffset` (GzipChunk.hpp:309-379) sets `outputSize = buffer.size()-nBytesRead` where `buffer = deflate::DecodedVector(ALLOCATION_CHUNK_SIZE=128 KiB)` allocated FRESH each `while(!stoppingPointReached)` iteration, then `buffer.resize(nBytesRead); result.append(std::move(buffer))` into a SEGMENTED `DecodedData` and loops with a NEW 128 KiB buffer. So rg DOES refill in 128 KiB chunks and NEVER reserves the whole chunk's output up front. DIS-14 conflated "the whole 128 KiB readStream buffer" with "the whole chunk". gzippy by contrast reserves `(compressed_span*8).max(4MiB).min(64MiB)` UPFRONT into ONE contiguous Vec (gzip_chunk.rs finish_decode_chunk_isal_oracle; segmented_buffer.rs `writable_tail_reserve`). The task premise (rg grows incrementally) is CONFIRMED; DIS-14's source claim is wrong.

**(1) IMPLEMENT (gated, OFF==identity).** `GZIPPY_ISAL_INCREMENTAL_GROWTH=1` routes the copy-free ISA-L decode through a new `decompress_deflate_from_bit_into_growable` (isal_decompress.rs) + `IncrementalOutSink` trait (impl on SegmentedU8): reserve a SMALL initial (`compressed_span*FACTOR`, floor 512 KiB) and GROW on `avail_out` exhaustion mid-decode (commit progress -> reserve more -> re-fetch tail -> resume `isal_inflate`) instead of returning `None`. ISA-L's streaming contract (internal `tmp_out_buffer` 32 KiB history) makes a fresh `next_out` after a Vec realloc byte-exact — rg relies on the identical property (fresh buffer each iter). Tunable `GZIPPY_ISAL_INITIAL_FACTOR` (default 4) / `GZIPPY_ISAL_GROW_MIB` (default 4). **Growth DISSOLVES DIS-14's sub-8 fallback constraint: even FACTOR=1 is 0-fallback** (under-reserve no longer falls back, it grows) — vs DIS-14's fixed factor-5/6/7 each forcing 1 pure-Rust fallback.

**(2) BUG CAUGHT (the all-dynamic-corpus blind spot).** First ON build was byte-exact on silesia (all-dynamic) but FAILED 4 in-tree routing tests on STORED/FIXED/btype01 fixtures ("parallel SM: output size mismatch"). Root cause: the growable path COMMITS bytes incrementally, but the caller has several post-decode `return Ok(false)` boundary-selection fallbacks (gzip_chunk.rs ~300/329, fire on stored/fixed where ISA-L records no usable boundary) that ASSUMED nothing was committed yet (the fixed path commits only at the very end) -> committed bytes left in -> pure-Rust fallback double-decodes. FIX: after a successful growable decode, `chunk.data.truncate(decode_start)` resets the LOGICAL length while the bytes stay PHYSICALLY in spare (Vec::truncate keeps capacity+contents) -> post-decode state byte-identical to the fixed-buffer path -> shared commit/CRC/boundary/fallback logic works unchanged. (Lesson: all-dynamic silesia dual-sha cannot exercise stored/fixed/btype01; the in-tree fixtures are load-bearing.)

**(3) MEASURED (frozen guest, bench-lock no_turbo=1 gov=performance runnable_avg<=1.5, mask 0,2,4,6 (T4) / 0,2,4,6,8,10,12,14 (T8), peak RSS min-of-N=7 via /usr/bin/time -v, perf stat -r 6 cpu_core PMU, vs rg 0.16.0; bin 908a9629; isal_chunks=14 isal_fallbacks=0 ASSERTED on EVERY arm; byte-exact: full lib suite 893 passed/0 failed OFF AND ON + corpus dual-sha OFF/ON T4/T8 = pin 028bd002…). Driver scripts/bench/{incr_growth.sh,_incr_growth_guest.sh} (NEW worktree tooling — FLAG TO STEWARD). Result REPRODUCED across 3 independent frozen runs.**

| metric (T4 / T8) | gz OFF (8x) | gz ON f4 | rapidgzip | verdict |
|---|---|---|---|---|
| dTLB-load-miss MPKI | 0.0623 / 0.0685 | **0.0366 / 0.0396** | 0.0466 / 0.0502 | ON_f4 **−41/−42%**, lands **BELOW rg** |
| dTLB-load-misses | 448k / 496k | 263k / 286k | 240k / 263k | ON ≈ rg |
| page-faults | 105k / 113k | 94k / 104k | 39k / 56k | ON **−8..11%** (still > rg abs) |
| peak RSS (KiB) | 211k / 296k | 216k / 297k | 153k / 223k | **NO drop** (f2 208k/308k, f1 224k/305k = WORSE) |
| wall (s, mean-of-6) | 0.696 / 0.432 | 0.688 / 0.426 | 0.468 / 0.352 | **TIE / no-regress** (ON marginally faster) |

**VERDICT — SPLIT (mechanism-backed):**
- **dTLB MPKI (the user's explicitly-UNMET "TLB half"): CLOSED.** ON factor-4 cuts dTLB-miss MPKI ~41-42% at BOTH T4+T8 and lands BELOW rapidgzip's, wall-neutral, 0-fallback, byte-exact. Mechanism: the 8x upfront reserve makes rpmalloc commit larger/sparser spans (DIS-14 saw the same "bigger reserve ADDS faults" at T4); a smaller dense buffer concentrates output writes -> better TLB coverage + fewer page-faults. **This is the on-goal footprint win on the TLB axis.**
- **peak RSS (the "+21-25% vs rg" half): NOT closed.** ON_f4 ≈ OFF; ON_f2/f1 WORSE. Mechanism (extends DIS-14's lazy-faulting finding from page-faults to RSS directly): peak RSS tracks TOUCHED/decoded bytes + Vec-realloc transients (old+new buffer) + rpmalloc span retention — NOT the lazy over-reserve (over-reserved pages are virtual, never resident). So shrinking the reserve CANNOT shrink RSS, and smaller initials (f2/f1) add regrow-realloc churn that RAISES peak RSS. **This REFUTES DIS-17's attribution "the +25% RSS is mostly the D1 8x output over-reserve"** — the RSS gap vs rg is the actually-touched working set (u16 marker buffers, per-chunk pipeline/window-map structs, D7 2-deep recycle hold), NOT the lazy output reserve.
- **factor=4 is optimal**: smallest single reserve that still covers silesia's ~3.3x decode ratio in one allocation (≈no regrow); f2/f1 regrow -> more spans -> erode the dTLB win + raise RSS.

**DECISION (rule 7a — a correct byte-identical change that HELPS is KEPT):** KEEP the gated change. It is byte-exact, wall-neutral, 0-fallback, and delivers the dTLB half of the user's cache-locality goal. RECOMMENDATION for the supervisor's Opus gate: flip the production default to ON with FACTOR=4 (closes the dTLB gap with zero downside); leaving it gated-OFF keeps production identical pending the gate. The RSS half is NOT reachable via the output-reserve lever — it must come from the touched working set (u16 marker machinery / pipeline buffers), a separate direction.

PROVENANCE: worktree .claude/worktrees/owner-isal-incremental-growth, branch owner/isal-incremental-growth (@ d56cb0f5 base). src/backends/isal_decompress.rs (`IncrementalOutSink` trait + `decompress_deflate_from_bit_into_growable`); src/decompress/parallel/segmented_buffer.rs (`IncrementalOutSink` impl on SegmentedU8); src/decompress/parallel/gzip_chunk.rs (`isal_incremental_growth` knob + gated branch + post-decode truncate). NEW bench tooling scripts/bench/{incr_growth.sh,_incr_growth_guest.sh} — FLAG TO STEWARD. Worktree submodules repopulated via `git submodule update --init` (worktree-add trap, same as DIS-21/22). Known-flaky `fd_vectored_write::early_reader_death` (pre-existing pipe-death hang on a loaded box, memory project_parallel_test_hang) `--skip`-ed; unrelated to this change. Clean: local leaked ~/.dotfiles/bin/timeout wrappers (+ssh children) reaped, guest lock-free no stray procs, neurotic no_turbo=0 watchdog=inactive (freeze released). OWES: supervisor's Opus disproof gate (Agent tool absent in owner env => self-disproof + pre-registered falsifier + frozen + sha-verified + freq-neutral-wall only).

---

## DIS-18 — "the T4 trough (gz LESS competitive at T4=0.90x than T8=1.01x) is an Amdahl crossover (H1: gz W larger, gz S smaller than rg), NOT a machinery defect (H2 pool-contention / H3 prefetch-starvation)" => H1 CONFIRMED, H2/H3 REFUTED [2026-06-09, owner/t4-curve @ d56cb0f5, bin b9eb0a733b4ccb6d, feature gzippy-isal]

> **LEDGER ANNOTATION 2026-06-13** | SCOPE: 2026-06-09, owner/t4-curve @ d56cb0f5, bin b9eb0a733b4ccb6d, neurotic guest 10.30.0.199 (no_turbo=1), gzippy-isal, silesia, T1..T9, N=13 interleaved | STATUS: **SCOPED** | CONFIDENCE: **S+W/T fit (moderate — model fit with r²=0.9959 for T3-T8; regression, not causal perturbation) + busy-fraction measurement (moderate — freeze-insensitive, but absolute walls differ from frozen parity runs)**. The H1 CONFIRMATION (W_gz > W_rg, S_gz < S_rg, crossover T*~7.5) is a model-fit result valid in the T3-T8 / fixed-chunk-count / silesia regime. SCOPE LIMIT: the S+W/T fit and the "gz keeps winning as T→∞" extrapolation are EXPLICITLY BOUNDED to the T3-T8 fixed-chunk regime (text says so). DIS-24 measured that the fixed-chunk premise breaks above T8 — the Amdahl crossover verdict stands ONLY in its banked box (silesia × T3-T8). NOTE: there are TWO DIS-18 entries in this file (see convention header); this is the second (Amdahl / T4-curve). The h2/h3 refutation (busy-fraction, false-sharing) is moderate-confidence. CONFIDENCE CALIBRATION: "H1 CONFIRMED" language is appropriate for a model-fit result within the stated scope; it would be over-confident if cited outside that scope.

**THE QUESTION (user):** why is gzippy-isal less competitive at T4 than T8? Fill the missing thread counts to see the curve shape and discriminate the lead hypothesis from machinery defects.

**METHOD.** Full T1..T9 curve, gzippy-isal (forced ParallelSM) vs rg 0.16.0, FROZEN guest (bench-lock no_turbo=1 gov=performance, uncore-pinned, runnable_avg 1.0-1.5), silesia.gz (raw 211,968,000 B), N=13 interleaved min-of-N, REGULAR-file /dev/shm sink, sha=028bd002…=OK EVERY cell, path=ParallelSM asserted every cell, bin b9eb0a733b4ccb6d. pin = one thread per P-core (even ids 0,2,..,14; T9 spills onto SMT sibling id 1 — flagged). Harness: scripts/bench/parity.sh (pin_mask extended to T2,3,5,6,7,9 + `--bypass`) + _parity_guest.sh.

**THE CURVE (gz wall / rg wall / ratio=rg/gz):**

| T | gz (s) | rg (s) | ratio | note |
|---|--------|--------|-------|------|
| 1 | 1.0245 | 0.9213 | 0.899 | forced-SM (serial-startup regime, DIS-15) |
| 2 | 0.9668 | 0.8353 | **0.864** | TROUGH (deepest) |
| 3 | 0.7033 | 0.6238 | 0.887 | |
| 4 | 0.5478 | 0.4933 | 0.901 | the user's T4 cell, reproduced |
| 5 | 0.4829 | 0.4518 | 0.936 | |
| 6 | 0.4362 | 0.4221 | 0.968 | |
| 7 | 0.3893 | 0.3900 | 1.002 | CROSSOVER (gz overtakes) |
| 8 | 0.3614 | 0.3653 | 1.011 | gz wins |
| 9 | 0.3864 | 0.3373 | 0.873 | chunk count 14->17->jumps; SMT-spill confound — gz REGRESSES T8->T9 while rg keeps scaling |

Monotonic ratio climb 0.864 -> 1.011 with a single crossover at T≈7 — the generic crossover signature. The deepest trough is actually T2 (0.864), not T4; T4 (0.901) sits on the rising limb. Scaling-vs-own-T1: gz 2.835x@T8 vs rg 2.522x@T8 (gz scales BETTER, starts BEHIND — exactly the asymmetry H1 predicts).

**FIT wall=S+W/T (scripts/analysis/sw_fit.py):**
- T3..T8 (the regime where S+W/T applies; drops the serial-startup T1/T2): r²=0.9959 (gz) / 0.9858 (rg). **W_gz=1607ms > W_rg=1188ms (+35%); S_gz=161ms < S_rg=217ms (−26%); crossover T*=7.49 ∈ (4,8). H1 CONFIRMED on all three clauses.**
- T1..T8 (incl. serial-startup): r²=0.82/0.86 (poor — T1->T2 is near-flat, ANTI-Amdahl, the DIS-15 serialization-bound startup phase, not steady parallel), so the full-range intercept is distorted (S_gz≈S_rg). The fit is only valid in the steady regime; reported both for honesty.

**CAUSAL DISCRIMINATOR (CPU busy-fraction = CPU%/(T·100), a within-process ratio => FREEZE-INSENSITIVE; /usr/bin/time -v, 3 reps, turbo-on runs — absolute walls differ from the frozen parity walls but the busy FRACTION is invariant):**

| | gz | rg |
|---|---|---|
| T2 busy-frac | 94% (188/200) | 91% (182/200) |
| T4 busy-frac | **91% (365/400)** | 82% (327/400) |
| T8 busy-frac | 72% (573/800) | 59% (472/800) |

- **At the T4 trough gz workers are 91% BUSY — NOT idle. This REFUTES H3 (prefetch-depth starvation) and H2 (pool-contention starving decode) as the trough mechanism: a starved pipeline shows LOW utilization; gz shows near-saturation.**
- **gz busy-fraction ≥ rg's at EVERY T** (91 vs 82 @T4; 72 vs 59 @T8). gz keeps its cores BUSIER than rg, i.e. gz's machinery/scheduler is at-least-as-efficient — independently corroborating S_gz < S_rg (rg hits its higher serial floor sooner => more idle => lower utilization, esp. at T8).
- rg `--verbose`: **Total Fetched = 17 chunks, CONSTANT at T2/T4/T8** (Prefetched 16 + 1 on-demand) => the chunk-count is fixed T1-T8 (H4 tail-wave/granularity REFUTED, as the plan predicted by arithmetic); decodeBlock≈0.47s constant (the W work). gz uses the same 4 MiB chunking => same ~17.
- T9 (chunk-adjustment fires at T≥9, count jumps): gz REGRESSES 0.3614->0.3864 while rg improves 0.3653->0.3373. The gz regression on MORE/smaller chunks = more serial marker-resolve/window-publish chain links = a RAISED floor S, which only hurts because gz is FLOOR-bound at high T — further (if SMT-confounded) corroboration that gz's high-T binder is S. (H5 kink present for gz; SMT-spill is a co-confound — flagged, not banked as clean.)

**VERDICT (engine-W vs machinery-defect): the T4 trough is ENGINE-W (Amdahl), NOT a machinery defect.**
gzippy carries a LARGER parallelizable decode work term W (1607 vs 1188 ms, +35% — the asm-bounded pure-Rust marker/inner-loop SYMBOL RATE, the standing MEMORY gap) but a SMALLER serial floor S (161 vs 217 ms, −26% — the publish-chain-overlap + post-process-alloc cuts, commits 85ad00a/0a40d5e/99ff098/0a3e9a3). Two such cost curves cross ONCE, at T≈7.5: at low T the wall is W/T-dominated => gz loses (T2/T4 trough); at high T it is S-dominated => gz wins (T7/T8). The MACHINERY (S, scheduler, worker utilization) is BETTER than vendor; the ENGINE (W) is worse. H2/H3 are refuted by direct measurement (gz 91% busy at T4, MORE utilized than rg, not less; chunk count constant). **LEVER: close W (inner-loop / marker-decode symbol rate) — the low-T headline; S is already at-or-below rg.** This matches MEMORY [project_pregate_placement_is_dominant_lever / engine-plateau] exactly.

**CALIBRATION (T1 bypass single-shot):** banked in DIS-15 (gz-singleshot 0.7659s = 1.197x rg). The `--bypass` harness extension here CORRECTLY caught that THIS binary (d56cb0f5) routes T1-production to ParallelSM, not IsalSingleShot — the `DecodePath::IsalSingleShot` route lives on the UNMERGED owner/t1-singleshot-route branch (finding for the steward: the DIS-15 routing lever is not in d56cb0f5).

PROVENANCE: worktree /Users/jackdanger/www/gzippy-t4-curve, branch owner/t4-curve (@ d56cb0f5 base). Harness: scripts/bench/parity.sh (pin_mask T2/3/5/6/7/9 + `--bypass`), scripts/bench/_parity_guest.sh (BYPASS mode), scripts/bench/_cpu_discriminator.sh (NEW — FLAG TO STEWARD), scripts/analysis/sw_fit.py (NEW). bin b9eb0a733b4ccb6d, gzippy-isal, frozen guest. Worktree submodules repopulated by local rsync from the main checkout (worktree-add empties submodule mountpoints — the same trap as DIS-21/22); a stray first rsync --delete-wiped the guest's vendor submodules, then RESTORED via the full-vendor re-sync (+ guest `git submodule update --init` at cleanup). NO Opus advisor available in owner env => self-disproof only; OWES the supervisor's Opus gate. Clean: leaked ~/.dotfiles/bin/timeout wrappers reaped; guest + neurotic lock-free (no_turbo=0, watchdog inactive, freeze released).

## DIS-24 — "the S-floor story HOLDS into the goal's 16+-thread regime (gz keeps winning as T→∞, smaller S); the T9 dip is the SMT-spill confound, not a machinery knee" => REFUTED. The curve TURNS OVER at T8: gz LOSES every cell T9..T32; the high-T binder is gz's T-PROPORTIONAL CHUNK-COUNT GROWTH (14→34), which raises BOTH W (flip_to_clean 12→31) AND S (publish-chain + per-chunk overhead) AND fb (0→1→2), while rg holds a FLAT wall. The "fixed-W" Amdahl premise is now MEASURED-contaminated above T8. [2026-06-09, owner/t4-curve worktree @ d56cb0f5, gzippy-isal bin b9eb0a733b4ccb6d (== DIS-18 binary), frozen guest 10.30.0.199 no_turbo=1 gov=performance runnable_avg=1.00, interleaved N=9, sha=OK every cell, path=ParallelSM asserted, vs rg 0.16.0]

> **LEDGER ANNOTATION 2026-06-13** | SCOPE: 2026-06-09, owner/t4-curve @ d56cb0f5, bin b9eb0a733b4ccb6d, neurotic guest 10.30.0.199 (no_turbo=1), gzippy-isal, silesia, T8..T32, N=9 interleaved, sha=OK | STATUS: **SCOPED** | CONFIDENCE: **wall measurement (strong for the T8-T32 loss curve; sha=OK, GZIPPY_VERBOSE per-cell counter capture)**. The BINDER NAMED ("T-proportional chunk-count growth") is an attribution-level inference (the chunk count grows and the loss worsens — correlated, not causally isolated). DIS-25 later found the chunk-count binder interpretation was WRONG (rg also grows chunks proportionally and has MORE chunks than gz at T24/32) — the chunk-count attribution should be read with DIS-25's correction. The CURVE SHAPE (gz loses T9..T32) is a direct measurement result and is not affected by DIS-25's attribution correction. NOTE: "rg holds a FLAT wall" at line 796 is PARTIALLY REFUTED by DIS-25 P1 (rg also grows chunk count; line 796 is corrected in DIS-25 P1).

**The amdahl-verdict-gate.md CLAIM-3(a) owed measurement — the goal's OWN target regime.** Method: a single frozen-snapshot multi-T topology sweep (scripts/bench/hicurve.sh + _hicurve_guest.sh, faithful extension of the parity spine: same env-scrub, fingerprint, corpus-sha oracle, frozen+quiet readback, REGULAR-FILE sink, interleaved best-of-N, sha-verify EVERY run, ParallelSM assertion) — build ONCE, freeze ONCE, loop the cells. i7-13700T topology VERIFIED on the guest: 8 P-cores w/ SMT (logical pairs 0-1..14-15) + 8 E-cores no-SMT (logical 16-23) = 24 logical. So the campaign's old "T16 = 0..15" mask is SMT-OVERSUBSCRIBED on 8 P-cores, NOT 16 physical cores — the exact confound the gate flagged.

CURVE (ratio = rg_wall/gz_wall, >1 = gz wins; gz forced-SM; per-cell counters):
| cell | P | pinning | gz_ms | rg_ms | ratio | verdict | chunks | isal/fb | flip_to_clean |
|------|---|---------|-------|-------|-------|---------|--------|---------|---------------|
| T8-Pphys | 8 | 8 P physical (no SMT/E) | 363 | 364 | **1.001** | TIE | 14 | 14/0 | 12 |
| T9-E | 9 | 8P + 1 E-core (no SMT) | 376 | 353 | 0.938 | TIE* | 19 | 18/**1** | 18 |
| T9-SMT | 9 | 8P + 1 SMT sibling | 386 | 344 | 0.890 | LOSS | 19 | 18/**1** | 18 |
| T10-E | 10 | 8P + 2 E | 406 | 347 | 0.855 | LOSS | 19 | 18/1 | 18 |
| T12-E | 12 | 8P + 4 E | 459 | 362 | **0.790** | LOSS | 23 | 22/1 | 20 |
| T14-E | 14 | 8P + 6 E | 487 | 374 | **0.768** | LOSS | 23 | 22/1 | 20 |
| T16-Ephys | 16 | 8P + 8E (16 PHYSICAL) | 439 | 378 | 0.861 | LOSS | 28 | 27/1 | 25 |
| T16-SMT | 16 | 8 P-cores × 2 SMT (the OLD T16 mask) | 368 | 336 | 0.912 | LOSS | 28 | 27/1 | 25 |
| T24-all | 24 | all 24 logical (P+SMT+E) | 399 | 355 | 0.889 | LOSS | 34 | 32/**2** | 31 |
| T32-oversub | 32 | -P32 on 24 logical (oversub) | 405 | 362 | 0.893 | LOSS | 34 | 32/2 | 31 |
(*T9-E is a TIE only under the harness's spread-margin; under the BINDING bar [MEMORY: TIE = ≥0.99x at EVERY T] 0.938 is a LOSS. Under the binding bar EVERY high-T cell T9..T32 is a LOSS — best is T9-E 0.938.)

**VERDICT — the goal's "16+ threads" regime is a LOSS for gzippy-isal.** The T-curve PEAKS at T7/T8 (≈1.00-1.01) and TURNS OVER: gz loses the entire T9..T32 range (0.94 → trough 0.77 @ T12-14 → 0.86-0.91 @ T16-T32). The S-floor "gz keeps winning as T→∞" story is REFUTED — it INVERTS past T8 exactly as the gate feared. gzippy-isal WINS/TIES rapidgzip ONLY in the narrow T7-T8 window.

**HIGH-T BINDER NAMED (from the per-T counters): gz's T-PROPORTIONAL CHUNK-COUNT GROWTH.** gz's chunk count (finish_decode) is NOT fixed — it scales with T: 14(T8)→19(T9)→23(T12/14)→28(T16)→34(T24/32). rg by contrast holds ~17 chunks CONSTANT (gate CLAIM 1) and a FLAT wall (336-378ms across T9-T32, already saturated/floored). Every chunk gz adds: (i) raises flip_to_clean = more speculative MARKER-decode = more engine-W (12→31, +158%); (ii) lengthens the serial marker-resolve/window-publish chain = a RAISED floor S (the DIS-18 mechanism, now measured at scale); (iii) raises fallback risk (isal_fallbacks 0→1→2, each ~7.5× re-decode W spike, DIS-14). So gz over-partitions as T grows while rg caps useful chunks — gz pays escalating per-chunk machinery for cores that (a) are E-cores (lower IPC than P) or (b) are SMT siblings (shared execution ports), neither delivering P-core-equivalent throughput. gz's wall RISES 363→487ms (T8→T14) then partially recovers as 28-34 chunks divide more evenly; non-monotonicity (T16 0.861 > T14 0.768) is tail-wave/granularity on the heterogeneous topology.

**TOPOLOGY CONTROL — the T9 dip DISENTANGLED (the gate's explicit ask):** T9-E (clean physical E-core, NO SMT spill) = 0.938; T9-SMT (SMT sibling of P-core 0) = 0.890. The dip is BOTH causes, in roughly equal parts: (1) a REAL MACHINERY KNEE — T9-E regresses from T8's 1.001 to 0.938 with ZERO SMT spill (driven by the chunk-count jump 14→19, the first isal_fallback firing 0→1, and the E-core's lower IPC); (2) an ADDITIONAL SMT-spill penalty — T9-SMT drops a further 0.938→0.890 from execution-port contention on the shared P-core. ⇒ DIS-18's "SMT-spill confound" at T9 is real but does NOT exonerate the machinery: the dip SURVIVES clean physical placement. The knee is the chunk-count machinery, not the SMT topology.

**The fixed-W Amdahl premise is now MEASURED-contaminated above T8 (the gate's CLAIM-1 GAP closed, NEGATIVELY):** the gate banked the S+W/T fit for T3-T8 only, owing fb=0 + flip_to_clean-plateau across the range. MEASURED: fb is NOT 0 above T8 (0→1→2) and flip_to_clean does NOT plateau (12→31). W is therefore T-DEPENDENT above T8 and the T3-T8 fit CANNOT be extrapolated to the high-T regime. The Amdahl crossover verdict (DIS-18) stands ONLY in its banked box (silesia × T3-T8 × the then-≈fixed-chunk regime); it does NOT and must not be quoted as predicting the high-T regime.

PROVENANCE: worktree /Users/jackdanger/www/gzippy-t4-curve, branch owner/t4-curve (@ d56cb0f5). NEW tooling (FLAG TO STEWARD): scripts/bench/hicurve.sh + scripts/bench/_hicurve_guest.sh (the one-snapshot multi-T topology sweep, with the verified i7-13700T mask table embedded + per-cell GZIPPY_VERBOSE counter capture + BUILD_ONLY mode). bin b9eb0a733b4ccb6d (== the DIS-18 binary; the guest binary was ALREADY gzippy-isal at d56cb0f5, incremental rebuild was a 0.10s no-op, fingerprint re-stamped). frozen guest, lock acquired+released cleanly (RESTORE VERIFIED no_turbo=0, watchdog inactive). NO Opus advisor in owner env => self-disproof only; OWES the supervisor's Opus gate. Clean: leaked ~/.dotfiles/bin/timeout wrappers reaped; guest + neurotic lock-free.

---

## DIS-25 — high-T A/B discriminator: chunk-count binder DEAD (measured), verdict = (B) heterogeneous-core work-distribution lever

> **LEDGER ANNOTATION 2026-06-13** | SCOPE: 2026-06-09, owner/high-t-discriminate @ d56cb0f5, bin b9eb0a733b4ccb6d, neurotic guest 10.30.0.199 (no_turbo=1, container live-expanded cores:16->24 then restored), gzippy-isal, silesia, T8-Pphys/T16-Ephys, N=7 | STATUS: **SUPERSEDED (primary verdict overturned by DIS-27)** | DIS-25's primary verdict "(B) heterogeneous-core work-distribution / prefetch-depth / out-of-order publish lever" is **directly overturned by DIS-27**: "DIS-25's (B) verdict … is OVERTURNED by direct worker measurement" (DIS-27 found gz's E-cores are BUSY/saturated at 72%, not idle/under-fed). The P1/P2/P3 measurement DATA from DIS-25 are valid (chunk-count binder DEAD per P1; per-core rate penalty equal on P/E per P2; rg extracts ~2x E-core wall benefit per P3 — these are measurements) but the INTERPRETATION (B: gz under-feeds E-cores) is wrong. The correct interpretation is DIS-27's: gz's E-cores are the BUSIEST resource, running ~1.59x more instructions per byte = engine-W amplified through the in-order pipeline. CONFIDENCE: P1/P2/P3 data moderate; the (B) verdict itself: SUPERSEDED. Do NOT cite the "(B) under-feeding" conclusion; cite DIS-27 for the high-T discriminator.

Scope: close the OWNER gate after DIS-24 — is the high-T LOSS (A) gz's per-chunk
engine-W amplified on low-IPC E-cores [same user-gated asm lever], or (B) a
separate heterogeneous-core SCHEDULING cost gz pays that rg avoids [new faithful
lever]? Method: new host-locked discriminator `scripts/bench/hitgate.sh` +
`_hitgate_guest.sh` (parity spine: env-scrub, fingerprint NO-rebuild on the
EXACT DIS-24 binary b9eb0a733b4ccb6d, corpus-sha oracle, frozen+quiet readback,
REGULAR-FILE sink, interleaved best-of-N=7, sha-verify EVERY run, ParallelSM
asserted). silesia.gz (raw 211,968,000 B). frozen guest, lock acquired+released
clean (RESTORE VERIFIED no_turbo=0, watchdog inactive). worktree
/Users/jackdanger/www/gzippy-hitgate @ d56cb0f5.

**GUEST CONFIG FINDING (flag to steward): the LXC guest 199 ("trainer") is
Proxmox `cores: 16` -> cpuset 0-15 = the 8 P-cores' SMT threads ONLY; E-cores
(cpu16-23) are NOT in the container's cpuset.** taskset -c 16 fails "Invalid
argument" from inside. DIS-24's E-core cells could only have run with the
container temporarily expanded to all 24 cores. For this run the container was
expanded live via `pct set 199 --cores 24` (cpuset 0-15 -> 0-23, applies live,
no restart), measured, then RESTORED to `--cores 16` + cgroup cpuset.cpus=0-15
(VERIFIED: guest nproc back to 16, taskset 16 fails again). Host i7-13700T, all
24 online; bench-lock KEEP_ALLOWLIST already includes 199.

### P1 — chunk-count binder is DEAD (direct measurement, both tools)
rg `--verbose` "Total Fetched" vs gz `finish_decode`, same -P, mask 0-23:

| -P | rg Total Fetched | gz chunks |
|----|------------------|-----------|
| 4  | 17 | 14 |
| 8  | 17 | 14 |
| 9  | 27 | 19 |
| 10 | 27 | 19 |
| 12 | 33 | 23 |
| 14 | 33 | 23 |
| 16 | 44 | 28 |
| 24 | 66 | 34 |
| 32 | 66 | 34 |

rg's chunk count GROWS with -P (17->27->33->44->66) exactly as
ParallelGzipReader.hpp:294-306 predicts — it is NOT "~17 constant" above T8
(REFUTES ledger line 796's "rg holds ~17 chunks CONSTANT"). And rg partitions
into MORE chunks than gz at every high-T cell (66 vs 34 at T24/32) yet stays
flat/winning. The "gz over-partitions vs rg" binder is doubly REFUTED: not only
does count fail to track ratio (DIS-24), gz actually partitions FEWER.

### P2 — single-core per-byte engine rate: engine-W deficit is REAL but NOT E-amplified
-P1 pinned, best-of-7, frozen (no_turbo=1 base-clock):

| tool | P-core(cpu0) | E-core(cpu16) | E/P |
|------|--------------|----------------|-----|
| gz   | 205.8 MB/s | 129.7 MB/s | 0.630 |
| rg   | 229.3 MB/s | 147.5 MB/s | 0.643 |

gz/rg per-core ratio = 0.898 (P) vs 0.879 (E) — essentially EQUAL. The E-core IPC
deficit (E/P ~0.63-0.64) is the SAME for both tools. gz pays NO extra per-byte
penalty on E-cores beyond the hardware IPC deficit shared by rg. => Discriminator
test for (A) "W amplified on E-cores" FAILS: there is no E-specific amplification;
gz is uniformly ~10-12% slower per core on BOTH core types.

### P3 — roofline / incremental-utilization: the high-T loss is (B), E-core under-feeding
Measured wall (best-of-7) vs single-core roofline (NPP*Prate + NE*Erate):

| cell | comp | gz wall | rg wall | ratio rg/gz | effGZ | effRG |
|------|------|---------|---------|-------------|-------|-------|
| T8-Pphys  | 8P+0E | 368ms | 378ms | **1.026 (gz WINS)** | 0.350 | 0.306 |
| T16-Ephys | 8P+8E | 333ms | 311ms | **0.934 (gz LOSES)** | 0.237 | 0.226 |

Decisive reads:
1. **gz WINS homogeneous T8 (1.026) despite a 10% single-core engine deficit** —
   gz's 8-P-core pipeline efficiency (0.350) beats rg's (0.306). If the loss were
   purely (A) engine-W (per-core ratio 0.90 on every core), gz would lose
   EVERYWHERE including T8. It does not. => engine-W is NOT the cause of the
   high-T-specific loss; it is MASKED at homogeneous T8.
2. **Incremental throughput from the 8 added E-cores:** gz wall 368->333 = +35ms
   (gains 61 MB/s of a 1038 MB/s E-core roofline = 5.9% captured); rg wall
   378->311 = +67ms (gains 121 MB/s of 1180 = 10.3% captured). **rg extracts ~2x
   more useful work from the same E-cores than gz.** That differential is exactly
   what flips gz from a T8 win to a T16-Ephys loss.

### VERDICT: (B) — a faithful high-T scheduling/work-distribution lever
The high-T LOSS is NOT (A) engine-W amplified on E-cores (per-core gz/rg ratio is
flat across core types; the W deficit is masked at homogeneous T8 where gz wins).
It IS (B): gz UNDER-FEEDS the slow E-cores in its parallel pipeline — captures
~half the incremental throughput rg gets from the same E-cores. Named lever:
**heterogeneous-core work distribution / prefetch-depth + out-of-order publish.**
Mechanism (hypothesis, source-grounded): gz's serial in-order publish-chain +
per-chunk flip_to_clean means a chunk landed on a slow E-core head-of-line-stalls
the in-order consumer, idling the fast P-cores behind it (cf. memory note
"confirmed-offset prefetch gap: high-T wall = head-of-line stalls"). rg COUNTERPART
= BlockFetcher prefetch cache (the --verbose "Prefetch Queue Hit" / Prefetched=16)
+ ParallelGzipReader work distribution + a deeper reorder window that absorbs the
slow-core straggler. This satisfies rule-7 (named rg counterpart, not a TIE).
Caveat (DIS-24 Q5 stands): silesia-only; rg's tuning corpus.

PROVENANCE: bin b9eb0a733b4ccb6d (DIS-18/DIS-24 binary, fingerprint MATCHED guest
src @ d56cb0f5, no rebuild). NEW tooling: scripts/bench/hitgate.sh +
_hitgate_guest.sh (worktree gzippy-hitgate, branch owner/high-t-discriminate). NO
Opus advisor in env => self-disproof only; OWES supervisor Opus gate. Clean:
container restored cores:16/cpuset 0-15, box thawed (no_turbo=0, frozen=[]), wd
inactive, leaked timeout wrappers reaped, local+guest+neurotic pgrep-clean.

---

## DIS-26 — the pre-registered DIS-6 item (i) consumer decompose + CAUSAL oracle, at HIGH-T (T16-Ephys/T24) => VERDICT (b): the high-T consumer is DECODE-WAIT-bound, NOT window-publish/post-process SERIAL-WORK-bound; the faithful lean-consumer port is REAL but a BOUNDED ~10-23ms ceiling (grows with chunk count), NOT the dominant binder. [2026-06-09, owner/dis26-consumer-decompose worktree @ d56cb0f5, gzippy-isal bin b9eb0a733b4ccb6d (decompose, NO rebuild) + bin 6e75f6dbff34f301 (perturbation, patched-rebuild, restored after), frozen guest 10.30.0.199 no_turbo=1 gov=performance runnable<=2.0, container LIVE-expanded cores:16->24 then RESTORED to cores:16/cpuset 0-15 (VERIFIED taskset16 fails), sha=OK every run, path=ParallelSM asserted, vs the DIS-24/25 banked walls]

> **LEDGER ANNOTATION 2026-06-13** | SCOPE: 2026-06-09, owner/dis26-consumer-decompose @ d56cb0f5, bins b9eb0a733b4ccb6d / 6e75f6dbff34f301, neurotic guest 10.30.0.199 (no_turbo=1, container expanded to 24 cores then restored), gzippy-isal, silesia, T8-Pphys/T16-Ephys/T24 | STATUS: **SCOPED** | CONFIDENCE: **FULCRUM timeline decompose (moderate — single instrumented capture per cell, self-test PASS) + causal perturbation for the publish-site delay (moderate — monotonic slope, freq-neutral sleep control, but ~1:1 response is partially tautological as noted in text)**. The "consumer serial-work is 2-7%" finding is well-supported (near-zero window_publish_clean self-time + decompose). The "lean-consumer ceiling = ~5-23ms growing with chunk count" is decompose-based (moderate). The "DECODE-WAIT-bound" label is vindicated by DIS-27 (E-cores BUSY decoding). NOTE: DIS-26 labels the decode-wait bucket "the engine" (an inference); DIS-27 then directly confirms it.

Closes the OWNER gate (high-t-B-gate.md "Highest-value next move") and the DIS-6
item (i) / DIS-25 named-lever that was owed a serial-work-vs-decode-wait decompose.

### Part 1 — DECOMPOSE (fulcrum_total on the wall-critical CONSUMER thread; --selftest PASS)
Method: GZIPPY_TIMELINE trace + GZIPPY_VERBOSE sidecar, env-unset production (NO
GZIPPY_SEED_WINDOWS), N=5 uninstrumented min-wall + 1 instrumented capture per cell,
sha-verified. fulcrum_total.py classifies the consumer (in-order, == the wall by
construction) leaf-time into WAIT (blocked on worker decodes) / OUTPUT (writev +
window_publish) / COMPUTE (serial post-process bookkeeping) / overhead. NOTE: the
tool's routing-guard REFUSE on window_seeded=2-3 is a FALSE POSITIVE here — these are
genuine production runs (no SEED env; flip_to_clean 12/25/31 dominant; isal_chunks
14/27/32), window_seeded=2-3 is the small production stream-start/sequential count,
NOT the GZIPPY_SEED_WINDOWS oracle (which seeds ALL ~16-17). The WAIT/OUTPUT/COMPUTE
split is computed independent of that guard.

| cell | P/mask | chunks | consumer span | DECODE-WAIT | OUTPUT(≈writev) | SERIAL post-proc(compute) | overhead |
|------|--------|--------|---------------|-------------|-----------------|---------------------------|----------|
| T8-Pphys  | 8 / 0,2..14            | 14 | 311ms | **66.1% / 206ms** | 30.8% / 96ms  | 1.7% / 5ms  | 1.3% |
| T16-Ephys | 16 / 0,2..14,16..23   | 28 | 269ms | **50.4% / 135ms** | 42.9% / 115ms | 3.6% / 10ms | 2.9% |
| T24-all   | 24 / 0-23             | 34 | 330ms | **54.2% / 179ms** | 35.7% / 118ms | 6.9% / 23ms | 2.9% |

THREE load-bearing reads:
1. **`consumer.window_publish_clean/marker` is sub-ms** (not in the top-20 self-time
   on ANY cell). The named "window-publish on the serial consumer" cost is negligible.
2. **`post_process.apply_window` (self 88/243/579ms) runs on the POOL, not the
   consumer** — the consumer's COMPUTE bucket is only 1.7-6.9%. gz ALREADY does the
   HEAVY post-process OFF the in-order consumer, faithful to rg. There is no large
   serial post-process on the consumer to "move to the pool" — it is already there.
3. The consumer wall is DECODE-WAIT (blocked on worker decodes: wait.block_fetcher_get
   72-113ms + ttp.rx_recv_block 42-70ms + future_recv) 50-66% + serial writev OUTPUT
   30-43%. The SERIAL-WORK the gate named (publish + post-proc on the consumer) is
   2-7%, and it GROWS with chunk count (5->10->23ms, T8->T16->T24 == the DIS-24
   T-proportional-overhead mechanism) but never dominates.

### Part 2 — CAUSAL perturbation (per-consumed-chunk delay at the consumer publish site)
Removal/perturbation oracle (CLAUDE.md rule 1): patched a gated per-chunk delay knob
(GZIPPY_DIS26_PUBLISH_DELAY_US; off==identity when unset, byte-exact every run) at the
consumer window-publish/post-process region (chunk_fetcher.rs:1665, before the publish
branch so it delays window publication + any downstream dependent worker). Rebuilt
gzippy-isal (bin 6e75f6db), swept at T16-Ephys (~28 chunks), N=7 min-wall, sha=OK,
then RESTORED the stamped binary. KIND=sleep (frequency-neutral, yields core) + KIND=spin
busy control:

  sleep:  0us=328ms  500us=343ms(+15)  1000us=361ms(+33)  2000us=398ms(+70)
  spin :  1000us=362ms  (== sleep 361ms => FREQUENCY-NEUTRAL, no turbo artifact)
  expected-if-fully-on-critical-path: +chunks*delay = +14 / +28 / +56ms

The wall response is MONOTONIC, ~1:1 with chunks*delay (only ~20% super-linear
compounding). Per rule 1 a proportional response => the SITE is on the critical path.
BUT per rule 3 (slow-down slope != speed-up ceiling) AND the high-t-B-gate tautology
warning: a ~1:1 response to injecting on the IN-ORDER consumer is largely the tautology
that the consumer IS the serial wall — injecting anywhere on it grows the wall. The
mild (~20%) compounding shows publish-LATENCY only partially gates downstream workers
(if workers were primarily publish-gated, compounding would be >>1.2x). The REMOVABLE
budget is therefore set by Part-1's decompose (the CURRENT serial-work bucket = 2-7%
= ~5/10/23ms), NOT by the injection slope.

### VERDICT: (b) DECODE-WAIT-bound, with a bounded (a) contribution.
- The high-T consumer is DOMINANTLY blocked on worker decodes (DECODE-WAIT 50-66% =
  the engine) + serial writev OUTPUT (30-43%). The named window-publish/post-process
  SERIAL-WORK is only 2-7% and the heavy apply_window is ALREADY on the pool.
- The faithful LEAN-CONSUMER port (move the residual publish/dispatch off the in-order
  consumer) is causally on the critical path but its CEILING is the serial-work bucket:
  ~5ms@T8 -> ~10ms@T16-Ephys -> ~23ms@T24, growing with chunk count. REAL and faithful
  (rg publishes off the in-order path) but BOUNDED — it cannot be the dominant high-T
  binder, cannot alone close the ~22ms T16-Ephys gz/rg gap, and shrinks toward 0 at the
  T7-T8 win window. This reconciles DIS-16 (T4 lean-consumer == TIE, serial work us-scale):
  the serial work is small at low-T and grows only modestly at high-T.
- The dominant high-T binder is DECODE-WAIT (the engine — the user-GATED asm lever;
  consistent with LEV-4 clean-rate 2.3x and DIS-24 chunk-count->engine-W growth) plus a
  separate serial-writev OUTPUT term (the DIS-5 output-overlap surface, refuted-as-path).

CAVEAT (DIS-24/25 Q5 stands, owed-separately, NOT done here): silesia is rg's tuning
corpus; a squishy-variety cross-check is owed before treating the high-T verdict as
corpus-general.

PROVENANCE: worktree /Users/jackdanger/www/gzippy-dis26 (branch owner/dis26-consumer-
decompose @ d56cb0f5). NEW tooling (FLAG TO STEWARD): scripts/bench/_dis26_capture_guest.sh
(decompose capture, byte-transparent) + _dis26_oracle_guest.sh (patch-build-sweep-restore
perturbation harness) + the gated chunk_fetcher.rs:1665 delay knob (worktree-only; the
guest patched binary was REBUILT then RESTORED to the stamped b9eb0a73 — guest source
verified clean, scratch removed). NO Opus advisor in env => self-disproof + pre-registered
slope/freq-neutral controls only; OWES supervisor Opus gate (do NOT claim advisor-vetted).
Clean: container restored cores:16/cpuset 0-15 (taskset16 fails, VERIFIED), box thawed
(no_turbo=0, frozen=[], wd inactive), local+guest+neurotic pgrep-clean (no gzippy/cargo/
rustc/bench-lock procs), leaked timeout wrappers none.

---

## DIS-27 — THE un-run discriminator (B-vs-b reconcile gate Q1): at the high-T LOSS cell (T16-Ephys = 8 P-core SMT threads + 8 E-cores) are the E-core WORKERS busy (engine-W, user-gated) or IDLE (under-fed = a faithful work-distribution lever)? => MEASURED **BUSY** — gz's E-cores are the BUSIEST resource (72% busy, MORE than gz's own P-cores), retiring ~1.59x MORE instructions than rg's E-cores for identical decoded bytes. VERDICT: **ENGINE-BUSY (terminal, user-gated). (B) under-feeding REFUTED.** [2026-06-09, owner/dis27-ecore-busy worktree @ d56cb0f5, gzippy-isal bin b9eb0a733b4ccb6d (== DIS-18/24/25/26 binary, fingerprint MATCHED, NO rebuild), frozen guest 10.30.0.199 no_turbo=1 gov=performance BENCH_LOCK=quiet runnable_avg=1.00, container LIVE-expanded cores:16->24 then RESTORED to cores:16/cpuset 0-15 (taskset16 fails, VERIFIED), sha=OK both tools every run, path=ParallelSM asserted, vs rapidgzip]

> **LEDGER ANNOTATION 2026-06-13** | SCOPE: 2026-06-09, owner/dis27-ecore-busy @ d56cb0f5, bin b9eb0a733b4ccb6d (== DIS-18/24/25/26 binary, fingerprint-matched, no rebuild), neurotic guest 10.30.0.199 (no_turbo=1, container expanded then restored), gzippy-isal vs rg 0.16.0, silesia, T16-Ephys, mpstat N=2 independent runs, perf stat N=2 | STATUS: **SCOPED** | CONFIDENCE: **mpstat %busy + perf per-PMU instruction count (moderate-to-strong)**. Two complementary instruments, both with instrument validation (frozen-idle baseline + T8-Pphys out-of-mask control — PASS). Sign and magnitude reproduced across two independent runs. The ~1.59x MORE instructions on gz's E-cores is the key finding; it directly reconciles DIS-25's ~1.91x "unexplained" 3/4 differential (DIS-27 text explains it is engine-W through in-order scheduling, not under-feeding). CONFIDENCE CALIBRATION: "ENGINE-BUSY (terminal, user-gated)" language is appropriate for the scope and evidence (per-core-type perf measurement, two-run reproducible); "terminal" means no further scheduling investigation is warranted given this binary/code state — re-measure if the engine changes significantly.

Closes the OWNER gate item DIS-26-could-not-label / B-vs-b-reconcile-gate.md Q1 ("DIS-26
measured the CONSUMER's decode-wait but NEVER the WORKER-pool idle, so it cannot label the
high-T DECODE-WAIT bucket 'the engine'"). This is the cheap decisive discriminator the gate
named: split per-core-type WORKER busy/idle at T16-Ephys.

### Method — per-core-type busy/idle, two complementary instruments, validated (Rule 4)
New host-locked discriminator `scripts/bench/_dis27_ecore_busy_guest.sh` (worktree
/Users/jackdanger/www/gzippy-dis27, branch owner/dis27-ecore-busy). The i7-13700T is a
HYBRID CPU with TWO PMUs: `cpu_core` (P-cores, cpu0-15 SMT) + `cpu_atom` (E-cores, cpu16-23)
— so the P/E split is HARDWARE-EXACT. mask T16-Ephys = 0,2,4,6,8,10,12,14,16,17,18,19,20,21,22,23.
  (1) **mpstat -P** per-CPU %busy (=100-%idle; time fraction; frequency-independent; the
      literal "idle fraction" the gate asked for), sustained 12s loop, grouped P vs E.
  (2) **perf stat -a** per-PMU global counters (cpu_core=P, cpu_atom=E): instructions +
      unhalted cpu-cycles = WORK DONE per core type (gap-free; an idle CPU retires ~0).
INSTRUMENT VALIDATION (both runs): frozen-idle baseline = P 3.7-4.3% / E 1.0-2.1% (~idle =>
the freeze isolates our workload); T8-Pphys CONTROL (E-cores NOT in the mask) = E-cores read
1.9-2.7% (~idle => no work mis-attributed to out-of-mask cores; no leakage). PASS.
NOTE (flag to steward): the LXC sees ALL 24 HOST CPUs via /proc — the bench-lock freeze
(pauses noisy neighbors) is what makes a per-CPU read attributable to our workload.

### THE MEASUREMENT (T16-Ephys, two independent runs — sign + magnitude reproducible)
| metric                         | gz run1 | gz run2 | rg run1 | rg run2 |
|--------------------------------|---------|---------|---------|---------|
| E-core mean %busy (mpstat)     | 71.9%   | 72.8%   | 53.2%   | 53.3%   |
| P-core mean %busy (mpstat)     | 67.0%   | 67.8%   | 63.5%   | 63.3%   |
| E-core instructions (perf)     | 4.612e10| 4.491e10| 2.877e10| 2.846e10|
| E-core instruction SHARE       | 38.4%   | 39.7%   | 34.7%   | 34.5%   |
| E-core IPC                     | 2.336   | 2.339   | 2.156   | 2.133   |

Decisive reads:
1. **gz's E-core workers are BUSY, not idle — 72% busy, the BUSIEST resource in the system
   (busier than gz's own P-cores at 67%).** In an in-order/work-stealing pool the slowest
   cores saturate first (always have a chunk to grind); gz's E-cores are OVER-subscribed, the
   long pole — the exact OPPOSITE of starved/under-fed. => the (B) under-feeding hypothesis
   (DIS-25 named lever: "gz UNDER-FEEDS the slow E-cores") is DIRECTLY REFUTED by measurement.
2. **gz retires ~1.59x MORE instructions on the E-cores than rg (4.5e10 vs 2.85e10) for the
   IDENTICAL decoded output** — the heavier engine-W (LEV-4 clean-rate 2.3x / DIS-18 +40%
   pure-Rust u16-marker decode) measured DIRECTLY on the E-cores. gz's E-core IPC (2.34) is
   HIGHER than rg's (2.13) => gz is NOT memory-/bandwidth-stalled on E-cores; it executes
   MORE instructions efficiently. Refutes the gate's "engine-W-under-bandwidth-contention"
   alternative too — it is plain heavier-W, not contention.
3. rg's E-cores are LESS busy (53%) yet (DIS-25) deliver ~2x the incremental wall benefit:
   rg's leaner engine = fewer instr/byte => each E-core chunk finishes FASTER, so rg's
   less-busy E-cores contribute throughput without becoming the in-order long pole.

### RECONCILIATION with DIS-25's 1.91x (the ~3/4 "parallel-only, not single-core-rate" gap)
DIS-25: 8 added E-cores drop gz wall 368->333 (-35ms) vs rg 378->311 (-67ms); rg extracts
1.91x; the flat per-core rate (gz/rg 0.879 on E) predicts only 1.14x; ~3/4 was "a
parallel-only feeding/contention property" — hypothesized (B). DIS-27 RESOLVES that ~3/4: it
is NOT under-feeding (gz's E-cores are the busiest resource, +1.59x instructions). It is the
HEAVIER ENGINE AMPLIFIED THROUGH THE IN-ORDER PIPELINE: gz's E-core chunks carry ~1.59x more
work, so each takes proportionally LONGER on the low-IPC E-core; the slow E-core chunk becomes
the long pole and head-of-line-stalls the in-order consumer (the E-core is BUSY decoding
during that stall, not idle), capping gz's incremental E-core wall benefit at ~half rg's.
This is EXACTLY the gate's "engine-W-under-(in-order-pipeline)-contention" reading, which the
gate itself stated "would still be the asm/clean-rate lever, INDISTINGUISHABLE at the consumer
without the worker-idle measurement" — DIS-27 supplies that worker measurement and the answer
is BUSY. So the 1.91x is engine-W (heavier per-chunk W) projected through scheduling, NOT a
separate feeding/dispatch lever.

### VERDICT: ENGINE-BUSY (terminal, user-gated). (B) under-feeding is REFUTED.
- The high-T E-core workers are BUSY decoding gz's heavier W (engine), NOT idle waiting for
  dispatch. DIS-26's "high-T DECODE-WAIT = the engine" label is VINDICATED at T16-Ephys: the
  consumer's decode-wait is the E-core workers grinding the heavier engine, not idle E-cores.
- DIS-25's (B) verdict (heterogeneous-core work-distribution / prefetch-depth / out-of-order
  publish "under-feeds the slow E-cores") is OVERTURNED by direct worker measurement.
- Mechanism (gate ask, source-grounded): gz's per-chunk worker engine = the pure-Rust u16
  MARKER-decode (src/decompress/parallel/chunk_fetcher.rs flip_to_clean :855 + the u16
  resolve/decode body in src/decompress/inflate/, DIS-18 ~57% of user instructions) is
  ~1.59x heavier on the E-cores than rg's worker (rg's leaner Block::read in
  vendor/rapidgzip/src/rapidgzip/GzipChunkFetcher.hpp + ChunkData.hpp, fed by
  src/core/BlockFetcher.hpp + Prefetcher.hpp). gz's dispatcher is NOT starving the E-cores
  — they are saturated; the lever is the ENGINE (user-gated asm), not the scheduler.
- This converges the B-vs-b contradiction onto engine-W-user-gated for the high-T regime, the
  ONE un-run measurement the reconcile gate (Q1/Q4) said was owed before "only user decisions
  remain." Still owed separately (NOT this gate): the per-chunk/ParallelSM-pipeline T1 removal
  oracle (var8-gate 124ms) and the squishy corpus-generality cross-check (DIS-24/25/26 Q5).

PROVENANCE: worktree /Users/jackdanger/www/gzippy-dis27 (branch owner/dis27-ecore-busy
@ d56cb0f5). NEW tooling (FLAG TO STEWARD): scripts/bench/_dis27_ecore_busy_guest.sh
(per-core-type worker busy/idle discriminator, READ-ONLY — no binary patch, no rebuild; reuses
the stamped b9eb0a73). Two independent runs, reproducible sign+magnitude. NO Opus advisor in
env => self-disproof + validated controls (frozen-idle + out-of-mask) only; OWES supervisor
Opus gate (NOT advisor-vetted). Clean: container restored cores:16/cpuset 0-15 (taskset16
fails, VERIFIED), bench-lock RELEASED (no_turbo=0, frozen=[], RESTORE VERIFIED, wd inactive),
guest scratch removed, local+guest pgrep-clean, neurotic shows only thawed neighbors
(ffmpeg/python/kvm — Plex et al. resuming), no bench-lock/timeout/gzippy orphans of mine.

---

## DIS-29 — THE owed DIS-23 force-regrow test + corpus-reframe-gate storm fix: does GZIPPY_ISAL_INCREMENTAL_GROWTH FIX the >8x fallback storm BYTE-EXACT, and recover the compressible-corpus low-T tax? => **YES at low-T, BYTE-EXACT, but it is a LOW-T-ONLY lever with a sub-8x REGRESSION — NOT a clean default-ON.**

> **LEDGER ANNOTATION 2026-06-13** | SCOPE: 2026-06-09, owner/isal-incremental-growth @ 153da9d1 (base d56cb0f5), bin 99d5758e2353543d, neurotic guest 10.30.0.199 (no_turbo=1), gzippy-isal, nasa/bignasa/ghcn/small/silesia, T1/T8/T16, N=5 interleaved per cell, sha=OK every arm | STATUS: **SCOPED** | CONFIDENCE: **strong for the byte-exact gate (dedicated force-regrow test, sha=OK), strong for the T1 recovery direction (isal_chunks 0→N confirmed, wall 0.554→0.662 on nasa), moderate for the parallel-T null result (N=5 only, but OFF≈ON at T8/T16 on bignasa — sign-stable)**. The sub-8x REGRESSION (ghcn −18%) is the key counter-finding; it prevents a clean default-ON. NOTE: file-order anomaly — this entry (DIS-29) appears BEFORE DIS-28 in the file; DIS-29 was added first chronologically. Read alongside DIS-23 for the full growth-lever picture. The storm IS reserve-overflow at low-T (confirmed first-hand); growth fixes the overflow chunks byte-exact and recovers T1 ~+20-30%, but (a) does NOT help parallel-T even on an 820MB 20-chunk corpus, (b) REGRESSES a sub-8x corpus (ghcn 7.77x) at T1 by ~18%, and (c) leaves residual fallbacks that are NOT reserve-overflow (window-absent bootstrap + ISA-L INVALID_BLOCK/SYMBOL on specific speculative-seed chunks, present in OFF too). [2026-06-09, owner/isal-incremental-growth worktree @ 153da9d1 (base d56cb0f5), gzippy-isal bin **99d5758e2353543d** (153da9d1 src + env-gated GZIPPY_STORM_DIAG instrumentation, behavior-identical with the knob unset), frozen guest 10.30.0.199 no_turbo=1 gov=performance BENCH_LOCK=quiet runnable_avg=1.00, T16=SMT mask 0-15, interleaved min-of-N=5, sha=OK EVERY arm EVERY cell, path=ParallelSM asserted, vs rapidgzip 0.16.0]

**(0) BUILD-IDENTITY TRAP CAUGHT (the guest.env "which-build-is-which" warning, live).** First build returned in 0.18s with bin_sha **b9eb0a73 == the DIS-28 NO-GROWTH binary** — a STALE warm-`target/` artifact cargo declined to recompile despite the 153da9d1 rsync (rsync -a preserved mtimes). A behavioral smoke test (nasa T1 OFF==ON, fb=5==5) initially looked like "growth does nothing" — it was the stale binary. A source edit forcing a real 37-38s recompile produced the true growth binary (99d5758e), and growth then worked. LESSON: assert a FRESH `Compiling gzippy` + new bin_sha, never trust a 0.18s "Finished" — exactly the trap the campaign warns of.

**(1) STORM MECHANISM — first-hand source-confirmed via per-Ok(false)-site diag (GZIPPY_STORM_DIAG, worktree-only, env-gated).** At T1 the nasa/bignasa storm is 100% `decompress_deflate_from_bit_into` returning None at the buffer-full site (out_pos==cap exactly; e.g. nasa 5/5 chunks overflow a ~35MB reserve = compressed_span×8) — the corpus-reframe-gate "reserve under-sizing at EXPAND_FACTOR=8" mechanism is CORRECT at low-T. BUT at T8+ the OFF fallbacks are MIXED: reserve-overflow PLUS ISA-L `ret=-1/-2` (ISAL_INVALID_BLOCK/INVALID_SYMBOL) after 78/367 bytes on specific chunks — those are un-decodable from their speculative window-seed and fall back in BOTH OFF and ON (NOT a growth defect; a separate speculation/seeding issue).

**(2) STORM FIX — CONFIRMED at low-T (isal_chunks 0→N, fb collapses).** Counter A/B (default factor 4, grow 4 MiB), T1 (cleanest — all chunks serialize through finish_decode): nasa **0/5 → 4/1**, bignasa(820MB,20chunks) **0/20 → 19/1**, small **0/2 → 1/1**; ghcn 7.77x (sub-8x control) **7/1 → 7/1 UNCHANGED** (no storm below the 8x threshold — clean control). Residual T1 fallback (1 chunk) is the window-absent bootstrap (gzip_chunk.rs:258), not reserve.

**(3) BYTE-EXACT — CONFIRMED, the owed DIS-23 force-regrow gate SATISFIED.** EVERY sweep arm sha=OK (OFF==ON==REF, gzip-oracle, every corpus×T). Dedicated forced-regrow run (factor=1, grow=1 MiB → many regrows/chunk): silesia **isal_chunks=14 fb=0 BYTE-EXACT** (~12 regrows/chunk × 14), nasa completing chunk BYTE-EXACT. The grow-mid-decode mechanism (commit→reserve→refetch→resume across realloc) is byte-exact on the real >8x regrow path silesia never exercised — DIS-23 Claim 1 EMPIRICALLY CONFIRMED.

**(4) WALL (frozen, min-of-5, rg/gz, >1=gz wins):**
| corpus (ratio) | T1 OFF→ON | T8 OFF→ON | T16 OFF→ON |
|---|---|---|---|
| small 10x (6MB) | 2.00→3.00 (0.03→0.02s) | 2.00→3.00 | 2.00→3.00 (startup-dom) |
| nasa 9.93x (205MB) | **0.554→0.662** (0.92→0.77s) | 1.000→0.964 | 1.042→1.000 |
| bignasa 9.93x (820MB,20ch) | **0.439→0.572** (3.71→2.85s) | 0.893→0.893 | 0.867→0.885 |
| ghcn 7.77x (250MB) | **0.941→0.800** (0.68→0.80s REGRESS) | 0.944→1.000 | 0.971→0.971 |

**(5) VERDICTS:**
- **T1 recovery: PARTIAL.** nasa T1 0.554→0.662 (+19.5%, the DIS-28 0.566 deficit), bignasa T1 0.439→0.572 (+30%). Real, but does NOT close to rg (residual = bootstrap chunk + ISA-L seed-error chunks + engine-W).
- **Parallel-T payoff: ~NONE.** Even bignasa (20 chunks ≫ 16 threads) shows T8/T16 OFF≈ON (0.89/0.87). At high-T most chunks are `finished_no_flip` clean continuations (decoded OUTSIDE the storm-prone finish_decode oracle), so only ~2-5 chunks ever take the overflow path regardless of total count. The storm's wall damage is T1/low-T-concentrated — confirmed on a large >8x corpus (DIS-28 point-3 owed measurement RESOLVED: the storm does NOT meaningfully tax parallel-T).
- **Sub-8x REGRESSION: ghcn 7.77x T1 0.941→0.800 (−18% wall 0.68→0.80s).** The always-small factor-4 initial under-covers a 7.77x corpus → ~1 regrow/chunk × 7 chunks of realloc+copy churn with NO storm to fix (OFF already 7/1). This is the cost of "always start small."

**DECISION / RECOMMENDATION for the supervisor's Opus gate:**
- KEEP the gated change (rule 7a): byte-exact, fixes the >8x low-T storm, delivers the DIS-23 dTLB win. The owed force-regrow byte-exact gate is now SATISFIED.
- **Do NOT flip the CURRENT knob (always-small factor-4 initial) to default-ON** — it regresses sub-8x corpora at low-T (ghcn −18%) and buys ~nothing at parallel-T. A clean default-ON is NOT justified.
- **Preferred production form (the corpus-reframe-gate's own alt): RETRY-ON-None-WITH-GROWTH.** Keep the 8x upfront reserve as the first attempt (so sub-8x corpora pay zero regrow churn — no ghcn regression), and on the buffer-full None, RETRY via the growable path instead of the 7.5x pure-Rust fallback. This strictly dominates both current arms: sub-8x identical to today, >8x rescued byte-exact. Owner-turnable; recommend implementing before any default flip.
- SEPARATE owed item (NOT this lever): the T8+ ISA-L INVALID_BLOCK/SYMBOL `ret=-1/-2` chunks (un-decodable speculative window-seed) — a speculation/seeding bug that caps the high-T compressible engine rate independent of reserve sizing.

PROVENANCE: worktree .claude/worktrees/owner-isal-incremental-growth @ 153da9d1. NEW bench tooling → Steward: scripts/bench/{storm.sh,_storm_build.sh,_storm_guest.sh} (parity/incr_growth spine: env-scrub, frozen+quiet readback, interleaved min-of-N, sha-verify EVERY arm, path-assert; builds nasa/small/ghcn from squishy + SYNTHETIC bignasa = nasa-raw×4 ~820MB/20chunks to expose parallel-T). WORKTREE-ONLY env-gated diagnostics (GZIPPY_STORM_DIAG, src/backends/isal_decompress.rs + src/decompress/parallel/gzip_chunk.rs) — STRIP before any merge; behavior-identical with the knob unset (measured bin 99d5758e). Corpora /dev/shm/corpusgen (RAM, /=5G tiny). Clean: host freeze RESTORE VERIFIED (no_turbo=0, watchdog inactive), guest+neurotic pgrep-clean, leaked local `timeout 540 ssh` wrappers (from the broken multi-word $SSH_JUMP through the dotfiles timeout shim — same shim that breaks lib_hostlock's freeze hop; acquired/released bench-lock.sh DIRECTLY instead) reaped. NO Opus advisor in owner env => self-disproof only; OWES the supervisor's Opus gate.

---

## DIS-28 — THE owed corpus-generality cross-check (DIS-24/25/26/27 Q5): is the engine-W / high-T-loss conclusion CORPUS-GENERAL or SILESIA-SPECIFIC (rg home-field)? => **SILESIA-SPECIFIC in its SHAPE: the "T8-win / T16-loss" curve does NOT generalize. The high-T LOSS WORSENS on near-incompressible (model 0.68), but REVERSES to a WIN on highly-compressible (nasa 1.05, small 1.74). The single-core engine-W deficit (T1 LOSS) is corpus-general in SIGN but its magnitude is corpus-driven (0.57-0.95), blown out by a FALLBACK STORM on compressible corpora (nasa/small: isal_chunks=0, ALL chunks fall back).**

> **LEDGER ANNOTATION 2026-06-13** | SCOPE: 2026-06-09, owner/corpus-general @ d56cb0f5, bin b9eb0a733b4ccb6d (fingerprint-pinned, NO rebuild), neurotic guest 10.30.0.199 (no_turbo=1, cores:16 never expanded), gzippy-isal vs rg 0.16.0, silesia/model/nasa/ghcn/small, T1/T8/T16-SMT, N=5 interleaved per cell, sha=OK every arm | STATUS: **SCOPED** | CONFIDENCE: **wall measurement (moderate — N=5 per cell; nasa/ghcn high-T wins sit on 7-12% spread, cited as TIE-or-WIN range; silesia reproduces DIS-24, instrument validated)**. The "SILESIA-SPECIFIC SHAPE" conclusion (T8-win/T16-loss curve) is directionally strong (4 corpora, 3 clear shape differences). The "engine-W ROOT is corpus-general" conclusion is attribution-level (consistent with DIS-27 per-core data but not a new causal perturbation). NOTE: file-order anomaly — this entry (DIS-28) appears AFTER DIS-29 in file order despite being the corpus-generality check that DIS-24/25/26/27 each flagged as owed; DIS-29 (storm fix) was added first. NOTE: PRE-FINDING at line 1273 explicitly states this binary has NO IsalSingleShot (using b9eb0a73 / d56cb0f5); the PRE-FINDING's routing claim is STALE for the current working tree (mod.rs now contains IsalSingleShot), but the MEASUREMENT DATA in the table are valid for the binary used (b9eb0a73 with ParallelSM at all T). Do NOT apply these corpus ratios to the IsalSingleShot-routed binary without re-measurement. [2026-06-09, owner/corpus-general worktree @ d56cb0f5, gzippy-isal bin b9eb0a733b4ccb6d (== DIS-18/24/25/26/27 binary, fingerprint-pinned EXPECT_BIN_SHA, NO rebuild), frozen guest 10.30.0.199 no_turbo=1 gov=performance BENCH_LOCK=quiet runnable_avg=1.00, container cores:16/cpuset 0-15 NEVER expanded (T16 = T16-SMT mask 0-15, no E-cores → no pct surgery), interleaved best-of-N=5, sha=OK both tools EVERY run EVERY cell, production routing (NO force) path-asserted+recorded per cell, vs rapidgzip 0.16.0]

Closes the squishy-variety cross-check the prior four gates each flagged "owed-separately, NOT done here; silesia is rg's tuning corpus." Method: new host-locked driver `scripts/bench/corpusgen.sh` + `_corpusgen_guest.sh` + `_corpusgen_build.sh` (faithful parity/hicurve spine: env-scrub allowlist, frozen+quiet readback, REGULAR-FILE sink, interleaved best-of-N, sha-verify EVERY run, per-cell GZIPPY_VERBOSE counters; pins the EXACT conclusion binary by bin_sha, NO rebuild — the conclusion's own binary tested across corpora). 4 squishy corpora spanning ratio 1.26x→10x and size 6MB→269MB + silesia as the home-field control.

### PRE-FINDING (routing, deterministic, unlocked): the task's "T1 routes to IsalSingleShot" premise is FALSE for this binary
On the `parallel_sm` build (gzippy-isal b9eb0a73), `classify_gzip` (src/decompress/mod.rs:170-188) routes EVERY single-member stream to **ParallelSM regardless of T or size** — there is NO IsalSingleShot path in this binary (that is a separate branch, owner/t1-singleshot-route). T1 = `ParallelSM threads=1`. So the engine under test at ALL cells is the pure-Rust/ISA-L-clean-tail marker-decode engine. Near-incompressible `model` did NOT route to StoredParallel (it has real Huffman blocks: chunks=51, flip_to_clean=49) — so the gate's "near-incompressible = mostly stored = little Huffman = different" hypothesis is REFUTED structurally: it routes to the SAME marker engine and exercises it HEAVIEST.

### THE CURVE (ratio = rg_min/gz_min; >1 = gz WINS; binding bar TIE = ≥0.99; T16=T16-SMT mask 0-15)
| corpus | ratio/content | T1 | T8 | T16 | chunks (T1/T8/T16) | isal/fb pattern |
|--------|---------------|-----|-----|-----|--------------------|-----------------|
| silesia | 3.11x mixed (home-field) | 0.904 LOSS | **1.022 WIN** | **0.918 LOSS** | 17/14/28 | isal=16-27, fb 0-1 |
| model | 1.26x safetensors (near-INCOMP) | 0.888 LOSS | **0.685 LOSS** | **0.677 LOSS** | 51/51/51 | isal=50-51, fb 0-1, **f2c=49** |
| nasa | 9.93x web log (HIGH-COMP) | **0.566 LOSS** | **1.044 WIN** | **1.048 WIN** | 5/3/3 | **isal=0/fb=ALL** (3-5) |
| ghcn | 7.77x numeric CSV | 0.945 LOSS | 1.009 TIE | 0.963 LOSS* | 8/2/3 | isal 1-7, fb 1 |
| small | 6MB 10x text (SMALL input) | **1.887 WIN** | **1.862 WIN** | **1.742 WIN** | 2/1/1 | isal=0/fb=ALL |
(*ghcn T16 0.963 is within the ~8% gz spread → LOSS-or-TIE marginal. nasa T8/T16 wins sit on a wide ~7-12% gz spread → robustly NOT-a-loss = TIE-or-WIN.)

### Decisive reads
1. **silesia REPRODUCES DIS-24** (T8≈1.02 win/tie, T16-SMT 0.918 ≈ DIS-24's 0.912) — instrument validated; the conclusion's curve is real ON SILESIA.
2. **The "T8-win/T16-loss" SHAPE is NOT corpus-general.** It only appears on silesia (and partially ghcn). On `model` gz LOSES every cell and WORST at T8/T16 (0.68); on `nasa`/`small` gz WINS at high-T. So "gzippy wins/ties the T7-T8 window, loses high-T" (DIS-24's headline) is a SILESIA ARTIFACT, not a corpus-general law.
3. **The engine-W single-core deficit (T1 LOSS) IS corpus-general in SIGN** (every large corpus loses T1: 0.566-0.945) — consistent with DIS-25's flat per-core 0.90 and DIS-27's +1.59x instr — BUT its MAGNITUDE is corpus-driven and dominated by a NEW term: the **FALLBACK STORM**. On highly-compressible corpora (nasa 10x, small 10x) the ISA-L clean-tail NEVER engages (isal_chunks=0); EVERY chunk falls back (the DIS-14 ~7.5x re-decode spike), crushing T1 nasa to 0.566. This is exactly the GOAL#2/DIS-14 "fixed-W premise breaks on flush-dense/odd corpora" warning, CONFIRMED corpus-triggered.
4. **Unifying mechanism (corpus-general): gz's deficit scales with COMPRESSED SIZE / CHUNK COUNT, not raw size; rg holds flatter.** model (213MB compressed → 51 chunks, f2c=49 = heaviest aggregate marker-W) = WORST loss 0.68. nasa/small (tiny compressed 20MB/0.6MB → 3/1 chunks) = the chunk-count amplifier is ABSENT and gz's LOWER fixed/startup overhead dominates → gz WINS high-T. This is the SAME DIS-24 chunk-count→engine-W binder, but it produces a high-T LOSS only when compressed-size is large enough to spawn many chunks. So engine-W is the corpus-general ROOT, but "high-T loss" is its expression ONLY on large-compressed corpora.
5. **Small inputs (6MB): gz WINS 1.7-1.9x at every T** — rg's fixed startup (~64ms) dwarfs gz's (~34ms); 1-2 chunks so parallelism is moot. The win/tie window framing inverts entirely for small files.

### VERDICT (for the OWNER gate)
**The engine-W ROOT is corpus-general (heavier per-chunk marker-W; gz loses single-core on every large corpus, worst where marker-decode is heaviest = model). But the DIS-24/25/26/27 "T8-win/T16-loss" CONCLUSION is SILESIA-SPECIFIC and does NOT generalize:** on near-incompressible data the high-T loss DEEPENS to 0.68 (engine-W is exercised HARDER, not bypassed — refuting the gate's stored-block hypothesis); on highly-compressible data the high-T loss REVERSES to a gz WIN (few chunks → chunk-count amplifier absent → gz's leaner startup wins); on small inputs gz wins everywhere. A previously-unledgered corpus-general term — the FALLBACK STORM (isal_chunks=0, all-fallback on ≥~8x-compressible corpora) — is the dominant single-core tax there and is NOT modeled by the silesia-tuned engine-W picture. The real-world picture therefore DIFFERS from the silesia headline: gzippy-isal's competitive position is a strong WIN on small + highly-compressible files, a TIE/marginal-loss on mixed text (silesia/ghcn), and a DECISIVE LOSS only on large near-incompressible files (model 0.68) — the inverse of where a "stored-block = easy" intuition would put it.

CAVEATS / not-done (partial-OK per owner discipline): N=5 (signs robust; nasa/ghcn high-T wins sit on wide 7-12% spread → read as TIE-or-WIN, not banked absolutes); T16 = T16-SMT (mask 0-15, 8 P-cores ×2 SMT) NOT T16-Ephys (E-cores) — chosen to avoid pct container surgery; the DIS-24 curve shows BOTH T16 variants are silesia-losses (0.912 SMT / 0.861 Ephys) so the corpus-generality question is answered by either, but the E-core-specific engine-W amplification (DIS-27) was NOT re-measured per-corpus. squishy corpora built with gzip -6 (real-world default), single-member.

PROVENANCE: worktree /Users/jackdanger/www/gzippy-corpusgen (branch owner/corpus-general @ d56cb0f5). NEW tooling (FLAG TO STEWARD): scripts/bench/{corpusgen.sh,_corpusgen_guest.sh,_corpusgen_build.sh} (corpus-generality driver + guest sweep + idempotent /dev/shm corpus builder; READ-ONLY on the binary — pins bin_sha, NO rebuild). Corpora in guest /dev/shm/corpusgen (ephemeral RAM; rebuildable from squishy URLs). NO Opus advisor in env => self-disproof + the silesia home-field control (reproduces DIS-24) only; OWES supervisor Opus gate (NOT advisor-vetted). Clean: bench-lock RELEASED (no_turbo=0, frozen=[], RESTORE VERIFIED, wd inactive), container NEVER expanded (nproc=16 throughout), leaked local timeout-wrappers reaped, local+guest+neurotic pgrep-clean.
