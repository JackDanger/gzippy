# Wall-parity scoreboard — the trustworthy progress signal

**Goal:** gzippy wall == rapidgzip wall (ratio **1.0×**) on the workload matrix.
**Instrument:** `scripts/whole_view.sh` section 1 — sha-verified, interleaved, best-of-7, self-tested.

## THE verdict rule (one line; the advisor-hardened version)
> **Progress ⟺ the sha-verified gzippy *ABSOLUTE* MB/s rose AND the interleaved best-of-7 ratio fell by MORE than its sample-spread, on a frozen host.**
> Everything else — refuted levers, "specificity", IPC, "it's a floor" — is *search*, not progress.

Why absolute too, not just ratio: a ratio can fall because **rapidgzip got slower** (box load), logging a fake gzippy "win" — the exact silent-instrument failure that burned this project. A row is **WIN** only if `gzippy_mbps ↑` AND `ratio ↓ > spread`. If ratio fell but gzippy abs is flat → **"rival regressed (load)" — NOT progress.**

## Row protocol (a row is INCONCLUSIVE unless all hold)
1. interleaved A/B (both binaries alternating, one run) — kills per-run jitter.
2. best-of-N, **N ≥ 7** (min, not mean — load only *adds* time).
3. sha-verified output on both sides.
4. **frozen host** if the claimed Δ < the inter-run spread (noise floor here ≈ **15%**, the 1.39–1.62 swing).
5. log the **sample spread**; Δ < spread ⇒ TIE, full stop.
6. self-test: a binary vs *itself* must read 1.0 ± spread, else the harness is lying → void.

## Trajectory (silesia-large, T8)

| date | commit / branch | gzippy MB/s | rapidgzip MB/s | ratio | spread | verdict |
|---|---|---|---|---|---|---|
| 05-30 | baseline (frozen) | ~1370 | ~2105 | **1.537×** | — | start |
| 05-30 | `69202e4` back-ref inline | **↑ (+13.7% same-run A/B)** | ~flat | **1.39×** | ~frozen | **WIN** (gzippy abs ↑ AND ratio ↓; controlled OLD-vs-NEW same run) |
| 05-30 | copy-collapse (reverted) | flat | flat | 1.40× | — | TIE (Δ<spread) |
| 05-30 | `feat/consumer-postprocess-pump` | flat | flat | 1.39× | in-noise | **TIE** — pump enqueued 0 tasks; premise falsified (stalls are on-demand decode of guard-rejected prefetches, not unpumped post-processing). 2nd TIE in a row ⇒ tripwire fired ⇒ re-localized. |

## 2026-05-31 (late) — EXECUTED: T1 fix + clean scaling sweep + coz infra (causal run pending workload fix)
- **Production path now `IsalParallelSM` at EVERY T (1-16)** — verified via GZIPPY_DEBUG. MIN_PARALLEL_SM_THREADS=0 (routing) + MIN_THREADS_FOR_PARALLEL 2→1 (engine floor, single_member.rs:31). T1 was ERRORING "input below parallel SM minimum"; now runs 1-worker+consumer, **byte-identical to rapidgzip (sha e114dd2)**, no hang. This is what makes the engine measurable at T1.
- **Clean FROZEN interleaved sweep (sha-verified, pinned):** T2 0.831×, T4 0.792× (both clean, spread ≤15%); T8/T16 too noisy (spread 24-36%, load crept 0.8→2.1). ⇒ gap is a **fairly UNIFORM ~20%** across thread counts, NOT a worsening cliff. **The earlier "gzippy wins P2 → scaling cliff" was a LOADED-BOX NOISE ARTIFACT** (confirmed). A uniform gap argues against a pure feeding/scaling story.
- **`fulcrum coz` causal run: infra stood up but NOT yet usable.** coz needs an UNSTRIPPED build (`CARGO_PROFILE_RELEASE_STRIP=false CARGO_PROFILE_RELEASE_DEBUG=line-tables-only RUSTFLAGS="-C dwarf-version=4"`, feature `pure-rust-inflate,coz`); progress point `chunk_emitted` + scopes `marker_bootstrap`/`clean_isal` are compiled in. BUT a ~0.4s decode is too short for coz to complete virtual-speedup experiments per run (60 reps → n_exp≈0). **NEXT: `coz run --end-to-end --fixed-speedup {0,25,50}` sweep on the marker_bootstrap line, OR a looped/long workload.** This is the gated entry to the real lever — do it before any code.

## 2026-05-31 — ⚠️ RETRACTION (Opus meta-audit): the "marker-decode-speed lever" below is the 4th PHANTOM
The head-to-head conclusion immediately below (window-absent marker decode SPEED = the lever) is **RETRACTED**. A meta-audit of the full history found:
- **It re-opens a measured-DEAD lever.** `x86-falsification-ledger.md:48-56` (FastBootstrap): a libdeflate-style u16 bootstrap decoder, 1.72–1.89× faster decode, byte-identical, produced a production wall **TIE** (N=11, 4 rounds, 3 host-frozen). "Decode RATE is wall-DEAD entirely." `lever-selection-gate.md:24`: decoder slice ceiling ~14%, CANNOT close the gap alone.
- **The 1.77× decode-CPU headline came from SINGLE un-interleaved traced runs** (one GZIPPY_TIMELINE each, loaded box, tracing perturbs timing) — violates this scoreboard's own N≥7/frozen/interleaved protocol. busy-CPU ≠ critical-path when workers are 37× overlapped.
- **`critpath`/`flow` attribution is BIASABLE** (analyst-chosen `preferred_blockers`); it manufactured 2 phantoms this session (decode-bias, scan_candidate umbrella). It is a hypothesis generator, NOT the verdict instrument.
- **The likely REAL lever is the SCALING CLIFF / consumer-feeding** (gzippy wins P2 0.93×, loses P4–P16; fill-factor 93→80%), in the 86% structural slice the gate says to attack FIRST.

**Corrected method (do this BEFORE any decode port):** the VERDICT instrument is `fulcrum coz` (CAUSAL virtual-speedup, empirical ∂wall±CI on the production parallel-SM binary, frozen host) — NOT static `--whatif`, NOT biasable critpath. Gate every lever: no code until a Coz run shows the wall is sensitive to that region with a CI that clears the gap. Build `fulcrum doctor` (path-assert + noise-gate + scaling-table + causal-verdict + ledger-check). The RLE-fill micro-opt (byte-identical, 19 tests pass) is NOT a claimed win — it targets the overlapped/wall-dead stage; do not promote without a causal+interleaved wall measurement.

## 2026-05-31 — FULCRUM HEAD-TO-HEAD (same instrument, both binaries) — lever localized [RETRACTED ABOVE]
Built `fulcrum flow` (committed fulcrum 6f920a8/8ee27df, 4 tests): per-stage WALL-CRITICAL vs TOTAL-BUSY (gap=slack), SERIAL/STARVED flags, `--whatif`. Then patched rapidgzip to emit the SAME Chrome-trace spans (scripts/rapidgzip_trace_patch, built `/root/gzippy/vendor/rapidgzip/librapidarchive/build-trace`) and ran the SAME tool on BOTH (T8, gzipcli-large 503MB, /dev/null).

**The instrument-consistent signal (both emit `worker.decode_chunk`):**
- rapidgzip worker decode busy = **1208ms** (8 workers) · wall 188ms
- gzippy worker decode busy = **2143ms** (8 workers) · wall 306ms
- ⇒ **gzippy's workers do 1.77× the decode CPU** (wall ratio 1.63× tracks it).

**Window-absent FRACTION is FAITHFUL (not the lever):** rapidgzip's own verbose reports "Replaced marker symbol buffers 31.25%"; gzippy decodes 31.97% window-absent. MATCH. The clean-window ARMING condition is a byte-for-byte port of vendor (deflate.hpp:1282-1284 ↔ deflate_block.rs:781-783), incl. the 64KiB/no-marker-ever clauses — vendor comments the 64KiB is deliberate. So reducing the window-absent fraction would be UNFAITHFUL and is dead.

**THE LEVER (measured, faithful-port-consistent):** gzippy's **window-absent MARKER decode SPEED** — pure-Rust ~160 MB/s (GZIPPY_VERBOSE body_rate) vs rapidgzip's unified decoder doing marker+clean at ~ISA-L class. Authorized by CLAUDE.md (inner Huffman reimplementation). Next cycle: speed the CONTAINS_MARKERS=true path in `read_internal_compressed`/`emit_backref_ring` (deflate_block.rs:1043-1188, 1668-1769) toward the clean ISA-L rate, emitting u16 markers. whatif: 1.6× bootstrap ⇒ −26% wall ≈ parity (Amdahl UPPER bound; advisor flags 209ms is straggler-gated so treat as ceiling not forecast).

**Three intermediate WRONG conclusions this session, each caught by cross-check (the discipline that worked):** (1) decode-bias in my own blame set → phantom; (2) `scan_candidate` umbrella double-counting decode → phantom "block-find 51%"; (3) "reduce the 32% fraction" → killed by rapidgzip's own 31.25% counter. Trust the SAME-instrument head-to-head, not one tool's number.

**Routing:** MIN_PARALLEL_SM_THREADS=0 (committed 4c876a0) — parallel-SM is the path optimized at every T, no libdeflate-one-shot confound at T1-3.

**Measurement gap (next "improve continuously"):** rapidgzip trace patch does NOT instrument consumer waits (no wait/recv spans) — "97% vs 0% consumer-wait" was an artifact, RETRACTED. Fix the patch's `wait.block_fetcher_get` site for a true wait-side head-to-head.

**OPEN:** parity not reached; T1-16 sweep + multi-archive + advisor final sign-off pending the inner-loop port.

## Convergence axis (replaces the gameable "kill count")
- Every candidate carries, **on paper before attack**, a `predicted_wall_ceiling ≥ remaining_gap` with its assumed fraction-on-critical-path. A lever that can't clear its own Amdahl bound is dead before it costs a work-stretch. (This would have pre-killed "marker-decode 82%" — it's 28% of WALL, ceiling < the 1.5× gap.)
- Convergence = the **measured critical-path bound is trending down across rows**, not the kill count.

## STALL tripwire (the one thing to watch)
> **2 consecutive *effortful* wall-A/Bs (build+measure spent) with Δ < spread ⇒ STALLED ⇒ mandatory RE-LOCALIZE (re-measure where the wall actually is) before attacking a 3rd lever.**
Unit = effortful work-stretches, NOT levers (a cheap paper-Amdahl kill doesn't tick it) and NOT wall-clock hours.

## Current candidate (advisor-validated, src-grounded) — CLOSE THE CHAIN INVARIANT
The pump TIE re-localized the wall: the consumer eats **on-demand heavy-chunk decode** because it **throws away range-valid prefetches** — `chunk_fetcher.rs:1152` demands exact equality (`max_acceptable_start_bit == next_block_offset`) where rapidgzip accepts a range (`encoded ≤ offset ≤ max`). The missing **chain invariant** blocks BOTH the prefetch-accept AND the pump's window cascade-publish (one gap, two dead levers — ledger OPEN LEVER 58-64). Lever: anchor `max_acceptable_start_bit` at the real found boundary on the fast path + guarantee the predecessor's end lands in `[encoded, max]`.
**BEFORE coding:** one traced run, confirm `PREFETCH_REJECT_BY_GUARD ≈ 4` (vs `on_demand`/`is_speculative` = late-prefetch instead). Reach ≈ 100–130ms (frontier/first-chunk irreducible). See `plans/refreshed-plan.md`.

## 2026-05-31 — `fulcrum vs` PER-THREAD-COUNT cross-tool comparison (the clear measurement)
Built `fulcrum vs A B` (committed fulcrum 260e154): span-by-span busy + wall-critical diff of gzippy vs traced-rapidgzip (same Chrome-trace vocab). Ran both at T1-16.
Wall ratio ≈ decode_chunk-busy ratio at EVERY T:
| T | wall | decode_chunk gz/rg |
|---|---|---|
| 1 | 1.14× | 628/545=1.15× |
| 2 | 1.67× | 1572/915=1.72× |
| 4 | 1.69× | 1736/991=1.75× |
| 8 | 1.65× | 1998/1126=1.77× |
| 16| 1.38× | 2443/1644=1.49× |
**Two separable causes (the targeted fixes):**
1. **Clean-decode ~1.15× slower (the T1 gap):** at T1 gzippy decodes via isal_stream_inflate 628ms vs rapidgzip 545ms — sequential, no speculation, yet 15% slower.
2. **Window-absent speculation tax = the dominant lever (T1→T2 jump 1.15→1.67×):** appears only at T≥2; `worker.bootstrap` (pure-Rust window-absent decode of the 31% markers) is 823-1142ms that rapidgzip does ~1.7× faster inside its decode. ~0.5× of wall.
gzippy-only overhead rapidgzip lacks: consumer.try_take_prefetched (803ms@T2), pool.pick (566ms@T16, grows with threads).
**Measurement gap:** rapidgzip trace patch doesn't instrument consumer WAITS (wall-crit=0 for rg) — busy half is decisive, blocking half not symmetric. Fix scripts/rapidgzip_trace_patch to wrap rapidgzip's getBlock wait for the complete picture.

## 2026-05-31 (late) — INDISPUTABLE finding: producer-side share ≠ wall-causality (5 causal TIEs)
Most powerful measurement built (awaited-chunk latency, ttp.rx_recv_block now carries awaited_offset, commit d0fa336): consumer 96.9% BLOCKED, blocks on 13/40 chunks, 100% of those = decode-LATENCY (decode already in-progress; 0% dispatch, 0% ordering, 0% window-wait). Sub-phase of the wait: bootstrap 54% / absorb_isal_tail 25% / clean-isal 21%.
**SELF-DISPROOF (the point):** that decomposition implies "speed bootstrap → wall drops," but TWO causal interleaved A/Bs say otherwise — FastBootstrap (decode-rate 1.7-1.9×) = TIE; backward-scan-skip (marker bookkeeping) = TIE (this session, T4/T8 frozen). So producer-side SHARE (busy/latency/awaited-keyed) is NOT wall-causality for this 8-worker overlapped pipeline. 5 TIEs prove it.
**coz BLOCKED in this LXC:** 2 attempts (non-end-to-end n_exp=0; end-to-end → only startup/runtime, no experiments) — coz needs perf-event PC sampling, restricted in the container. The standard causal tool does not function here.
**Indisputable causal test that DOES work here:** deterministic slow-injection A/B (env-gated delay ∝ bootstrap's own time; measure interleaved wall delta). 2×-slower-bootstrap → ~0 wall delta ⇒ bootstrap conclusively wall-dead. NEXT.
**Implication:** parity is unlikely from bottleneck-hunting; the structure (speculate-then-resolve) is the suspected cause. The decisive experiment is the known-window SINGLE-DECODE oracle (advisor) — also the stated vision (unify bootstrap+marker decode, decode once). Build that.

## 2026-05-31 — ★ CAUSAL BREAKTHROUGH (slow-injection, indisputable): bootstrap IS the lever
coz is blocked in-container, so used a DIRECT causal perturbation: GZIPPY_SLOW_BOOTSTRAP=N spins N% of the bootstrap's own time (byte-identical pure delay). Frozen, interleaved, sha-verified, T8:
| bootstrap slowdown | wall | Δwall |
|---|---|---|
| +0% | 0.345s | — |
| +50% | 0.378s | +9% |
| +100% | 0.447s | +30% |
| +200% | 0.543s | +57% |
Monotonic & proportional ⇒ the window-absent bootstrap CAUSALLY gates ~30% of the wall (slope ~0.3× wall per 1× bootstrap-time). This is a direct perturbation of EXACTLY the bootstrap — not biasable attribution — so it is indisputable.
**OVERTURNS the 5-TIE "decode is wall-dead" doctrine.** The prior TIEs (FastBootstrap, backward-scan-skip) were too-small/unrealized bootstrap speedups (few-% change invisible under the 10% noise floor); a 50-200% perturbation is unmistakable. Direct causal test > indirect optimization A/Bs.
**LEVER (causally confirmed, not a phantom):** speed the window-absent bootstrap Huffman decode (pure-Rust ~160 MB/s → rapidgzip-class). Predicted ~1.7× faster ⇒ ~-15-20% wall ⇒ most of the T8 gap. This IS the inner-Huffman reimplementation (hand-tuned Rust, drop the slow path) — the stated vision. Method that worked: when coz is unavailable, slow-inject a region proportional to its own time and measure interleaved wall response (linearity = causality).

## 2026-05-31 — ADVISOR VERDICT on the causal breakthrough (adversarial sign-off)
SIGNED OFF (robust): **the window-absent bootstrap is ON the critical path, NOT overlapped slack** — slowing it 2× → +30% wall is impossible for off-critical-path work (3× the noise floor). This FALSIFIES the "decode is wall-dead / 5-TIE" doctrine. The earlier meta-audit's "bottleneck-hunting can't help" rested on that doctrine and is weakened.
NOT signed off (the lever ceiling): "speeding bootstrap = -15-20% wall" is NOT established because:
1. Slow-down slope ≠ speed-up ceiling (asymmetry: speeding helps only until the consumer knee binds). The -15-20% is linear extrapolation through an unlocated knee.
2. busy-spin TURBO-DEPRESSION confound — 8 spinning cores can depress all-core turbo, inflating the +30% (it's an UPPER bound). Need a frequency-neutral delay or per-core MHz logging.
3. FastBootstrap TIE may ALREADY be the speed-up result (if its 1.7× was real and hit the knee). Need forensics: did it actually move production bootstrap_dur_us, interleaved?
DECISIVE next experiment = the bootstrap-REMOVED upper bound: `drive_clean_window_oracle` (chunk_fetcher.rs:283, GZIPPY_CLEAN_WINDOW_ORACLE=1). **BUT it is BROKEN** — produces empty output (sha e3b0c442 = empty-string), matching the memory's "broken clean-window oracle" warning. REPAIR IT, then oracle-wall vs production-wall = the speed-up ceiling. If oracle ≈ production → lever ceiling ~0 (structural change needed); if oracle ≪ production → bootstrap speed is the lever.

## 2026-05-31 — DISPROOF SURVIVED: bootstrap criticality is real (turbo confound refuted)
Attempted to disprove "bootstrap on critical path" via the advisor's frequency-neutral control: re-ran the +100% injection as a SLEEP (yields the core, cannot depress turbo) instead of a busy-spin. T8 frozen interleaved sha-verified:
- base 0.342s · spin+100% 0.446s (+30%) · sleep+100% 0.435s (+27%).
The sleep delta (+27%) ≈ the spin delta (+30%), both ≫ the 12% noise floor ⇒ the turbo-depression confound is REFUTED; the wall response is genuine bootstrap criticality. SIGNED-OFF claim stands: the window-absent bootstrap is ON the critical path (~0.27-0.30× wall per 1× its time), NOT overlapped slack. "Decode is wall-dead" is FALSE.
STILL OPEN (not disproven, not confirmed): the speed-up CEILING — slow-down slope ≠ speed-up gain; needs the bootstrap-removed oracle (currently broken, emits empty output) repaired.
PROCESS encoded into CLAUDE.md (replacing the falsified "decode wall-dead / structural-slice / 14%-ceiling" conclusions): perturb-don't-attribute; frequency-neutral control; slow-slope≠speed-ceiling; validate-instrument-first; disproof-driven; frozen/interleaved/N≥7/sha.

## 2026-05-31 — oracle repair diagnosis (the gating experiment for the speed-up ceiling)
`drive_clean_window_oracle` (GZIPPY_CLEAN_WINDOW_ORACLE=1) fails in PHASE A (sequential dict-build) with `InflateFailed(ResumableInflate("Stored block: len=0x0 nlen=0x0"))`. Root cause: span starts come from `seed.keys_snapshot()` (speculative pass-1 published window keys), and at least one key is NOT a confirmed deflate block boundary, so `decode_chunk_isal` starting there misreads. REPAIR: seed Phase-A spans only at CONFIRMED block boundaries (validate each start decodes a legal block header before chaining), or derive boundaries from a trusted sequential first pass. Once repaired, Phase-B pass2 wall vs production wall = the speed-up CEILING (the one number that says whether speeding the bootstrap can reach parity, since slow-down slope ≠ speed-up gain). This is the gating experiment; until then the bootstrap speed-up ceiling is UNKNOWN (open, not concluded).

## 2026-05-31 — oracle CORRECTNESS fixed; DISPATCH confound found (ceiling still not clean)
Repaired oracle Phase A (self-derive boundaries from decoder end-bit, commit 14d4c5f): now byte-identical (sha e114dd2 = ref). BUT pass-2 wall = 0.409s / 1233 MB/s — SLOWER than production 0.345s. Not "speculation-removal hurts": 1233/8 ≈ 154 MB/s/worker ≈ single-worker clean rate ⇒ the oracle's Phase-B dispatch (hand-rolled std::thread::scope WAVES of pool_size, chunk_fetcher.rs:372) gives only ~1.2× effective parallelism (wave barriers serialize; stragglers stall the whole wave). The doc comment "reuses the entire production pool/consumer" is FALSE. So the oracle wall is DISPATCH-limited, not the clean-decode ceiling. REMAINING FIX: route Phase B through the production pool+consumer with pre-seeded windows (WindowMap dict per span), then pass-2 wall = the true clean-parallel ceiling = the bootstrap speed-up bound. Process win: instrument validated (correctness ✓), confound identified (dispatch), next fix specified.

## 2026-05-31 — ★★ ORACLE CEILING (dispatch fixed): speculation is ~61% of wall; clean engine beats rapidgzip
Fixed oracle Phase-B dispatch (atomic work-queue, no wave barriers, commit b2e75e9). Frozen T8, byte-identical (sha e114dd2):
- gzippy CLEAN-parallel ceiling (known windows, NO bootstrap/markers): **0.110s / 4584 MB/s**
- rapidgzip: 0.197s
- gzippy production (with speculation): 0.279-0.293s
⇒ The clean decode engine is NOT the bottleneck — it is 2.5× faster than production and 1.8× faster than rapidgzip. The ENTIRE gap (and more) is the window-absent SPECULATION/bootstrap overhead (~0.17s ≈ 61% of gzippy's wall). Combined with the slow-injection (bootstrap ON the critical path, disproof-survived), this is the complete causal picture: **the lever is the window-absent bootstrap; it has ~2.5× headroom.** The clean path (decode_chunk_isal) is excellent; the speculation is the cost.
Caveat (honest): the oracle's 0.110s assumes windows are FREE (Phase A computes them, untimed). You cannot get windows free in single-member — you must speculate (as rapidgzip does, 31% markers) OR decode sequentially. So 0.110s is the UPPER BOUND on "decode-once-with-windows", not directly achievable; it bounds the win from making speculation cheaper/faster. rapidgzip speculates too yet hits 0.197 — so gzippy's speculation is specifically ~1.4× less efficient than rapidgzip's, and that inefficiency (not the clean engine) is the closable gap.

## 2026-05-31 — ADVISOR CORRECTION: the "2.5× headroom" is a FICTION; real target = 1.4× window-absent-path gap
Adversarial advisor PARTIAL sign-off:
- SIGNED OFF (cause): the gap is ENTIRELY the window-absent/speculation path; the clean decode engine (decode_chunk_isal, 4584 MB/s) is NOT the bottleneck. The slow-injection (+ frequency-neutral SLEEP disproof) is bulletproof: bootstrap is on the critical path.
- STRUCK (fiction): "2.5× headroom / clean ceiling 0.110 < rapidgzip 0.197" — the oracle's Phase A (window computation) is UNTIMED and UNREACHABLE in single-member (you can't have windows without producing them = the very speculation). 0.110s is an ENGINE ceiling, NOT an achievable-wall target. Comparing free-window-oracle to with-speculation production is a category error.
- REAL TARGET (defensible): rapidgzip ALSO speculates (31% markers) and hits 0.197 vs gzippy 0.279 ⇒ gzippy's window-absent path is ~1.4× less efficient. Close THAT 1.4×, not the fictional 2.5×.
- INNER-LOOP PORT NOT YET AUTHORIZED: the 1.4× may be in marker RESOLUTION / apply_window / consumer-copy (structural), not the Huffman decode loop — memory ("copies wall-neutral", "wall IS consumer critical path") warns it's structural. backward-scan-skip TIE hints bookkeeping isn't it, but that's indirect.
- GATING MEASUREMENT before any multi-session port: slow-inject INSIDE the marker-decode inner loop ALONE (decode_huffman_body_resumable / marker emit), not the whole bootstrap. Wall responds ⇒ inner-loop is the lever, port justified. Wall flat ⇒ lever is structural (consumer/marker-resolution convergence to rapidgzip), inner-loop port would be wasted.

## 2026-05-31 — decode-vs-resolve discriminator SATISFIED (injection is worker-scoped) ⇒ inner-loop lever confirmed
The advisor required: prove the 1.4× is in the DECODE, not the resolution, before authorizing an inner-loop port. It already is:
- The GZIPPY_SLOW_BOOTSTRAP injection sits in `decode_chunk_marker_bootstrap_then_isal` (gzip_chunk.rs:890), the WORKER decode, BEFORE chunk handoff. Marker RESOLUTION (`apply_window`/`narrow_u16_to_u8`) runs CONSUMER-side (chunk_fetcher.rs:2331), downstream of the injection. So the +30% (sleep-disproof-survived) slows ONLY the worker-side window-absent DECODE — the resolution is excluded by construction.
- Within that decode, the marker-ring bookkeeping is NOT the cost (backward-scan-skip + RLE-fill both TIED). ⇒ the cost is the window-absent Huffman symbol decode itself.
**DIAGNOSIS COMPLETE (causal, advisor-aligned):** the gzippy→rapidgzip gap is the worker-side window-absent Huffman DECODE, ~1.4× less efficient than rapidgzip's equivalent. NOT the clean engine (4584 MB/s, excellent), NOT the consumer resolution, NOT scheduling. Target: close the 1.4× by reimplementing the window-absent marker-emitting Huffman decode in hand-tuned Rust (the authorized inner-loop work + the user's stated vision). This is the multi-session build the diagnosis now justifies.

## 2026-05-31 — ★ LEVER SHARPENED: gzippy has TWO pure-Rust decoders; window-absent uses the SLOW one
Verified the oracle's fast clean path is PURE-RUST (not FFI): the pure-rust-inflate build's `IsalInflateWrapper` wraps ResumableInflate2 (gzip_chunk.rs:25-26 "pure-rust-inflate build does NOT use real ISA-L"). So the 4584 MB/s clean ceiling is a pure-Rust decoder.
THE TWO DECODERS:
- CLEAN tail: `IsalInflateWrapper`/ResumableInflate2 — FAST (4584 MB/s, oracle).
- WINDOW-ABSENT bootstrap: `bootstrap_with_deflate_block` → `deflate_block::DeflateBlock` (MarkerSink) — SLOW, and it is the ~1.4×-to-rapidgzip / ~61%-of-wall critical path (slow-injection + oracle + sleep-disproof).
⇒ The lever is NOT "speed the Huffman loop" in the abstract — it is that the window-absent path runs a DIFFERENT, slower decoder (deflate_block) than gzippy's own fast clean decoder. **UNIFY THEM:** route the window-absent decode through ResumableInflate2's fast machinery WITH u16 marker emission (or port its speed into deflate_block). This is EXACTLY the user's vision ("unify the bootstrap and the parallel marker decode"). Concrete, vision-aligned, and the diagnosis (causal + advisor-signed cause) justifies it. NEXT SESSION'S BUILD: make deflate_block's window-absent decode match ResumableInflate2's rate; measure interleaved wall (target: close the 1.4× to rapidgzip).

## 2026-05-31 — FASTLOOP transplant MEASURED + FALSIFIED (mechanism wrong, lever still right)
Did the WHOLE change (3-agent flow: design→implement→measure, per user directive to stop decomposing). Implemented the designed lever: transplant ResumableInflate2's two-tier FASTLOOP + branchless refill into the window-absent decoder (deflate_block DYNAMIC arm). Build clean, byte-identical (deflate_block 19/19, routing 25 incl byte-perfect T4-on-24MiB).
RESULT: wall TIED at T4/8/16 (frozen interleaved sha-verified vs pre-FASTLOOP + rapidgzip). DISCRIMINATOR: bootstrap body_rate 157→160 MB/s (+2%, noise) ⇒ the FASTLOOP did NOT realize a decode speedup. **FALSIFIED: the loop structure (yield-elide/branchless refill) is NOT the window-absent decoder's bottleneck.** Reverted (don't ship tied complexity).
REFINED TARGET (the decode-rate gap is real — clean 573 MB/s/worker vs window-absent 157 — but NOT the loop): candidates per Agent1's code analysis = (i) TABLE PRIMITIVE — window-absent uses ISA-L 10-bit IsalLitLenCodePure; clean uses libdeflate 11-bit LitLenTable + multi-literal chaining (the proven-faster primitive); (ii) u16 ring store width (2× bytes + phys/pos two-counter) vs clean's flat monotonic u8; (iii) per-read() 64KiB re-entry (RING_SIZE cap) re-running setup. NEXT mechanism to implement+measure: swap the window-absent decode's table primitive to the clean path's LitLenTable+multi-literal, OR widen the store path. The 3-agent design→implement→measure loop is the method; FASTLOOP was the first measured-falsified mechanism.

## 2026-05-31 — LAYERED unification + consumer-copy elim: granular result (decode faster, wall marginally better, kept)
Per user directive (don't work incrementally; layer correct changes; don't revert; re-measure WHOLE; if not strictly better get granular + reason). Layered on feat/fulcrum-causal-sweep @5ba2ddf: (1) FASTLOOP transplant, (2) LitLenTable+multi-literal core (replaces ISA-L IsalLitLenCodePure), (3) consumer-serial window-publish copy elim (from_owned_none, kills the 2nd 32KiB to_vec), (4) fuse threshold 128->16KiB. ALL byte-identical (sha e114dd2; deflate_block 19/19; routing 25 incl byte-perfect T4/24MiB).
GRANULAR (fulcrum vs, baseline gz_pre vs all-layered, T8 single-trace):
- wall 294.7->276.7ms (-6% this trace; frozen INTERLEAVED A/B = ~tie to +1-5%, within 13-28% spread).
- BETTER: worker.block_body 1022->877ms, worker.bootstrap 1031->886ms (decode -145ms busy — the unification DID speed the decode; the earlier 124->125 body_rate was a loaded-box artifact). consumer.iter -18ms, try_take_prefetched -14ms.
- WORSE (small): post_process.task -12.5ms, isal_stream_inflate -9ms.
REASONING: decode is now faster but OVERLAPPED across 8 workers, so the wall gain is small + noise-masked. The binding constraint stays the in-order consumer waiting on per-chunk latency. All changes correct+kept (layer, don't revert). Next lever (system reasoning): the consumer serial path / per-chunk latency — keep layering obvious consumer-serial wins (background agent's findings #2,#4,#5 remain) + the still-untried output-store path (u16 ring store width vs clean u8, candidate ii — the decode-vs-store split showed store is now the larger share since decode is at clean-table parity).

## 2026-05-31 — CUMULATIVE LAYERED stack moves the wall (frozen N=11, vs baseline gz_pre)
5 layered byte-identical changes (FASTLOOP + LitLenTable core + consumer window-publish copy-elim + fuse 128->16KiB + back-ref word-copy @43fdc1c). All kept (layer, don't revert). sha e114dd2.
| T | baseline MB/s | layered MB/s | gain | ratio vs rapidgzip |
|---|---|---|---|---|
| 4 | 1122 | 1224 | +9% (>spread) | 0.753 -> 0.821 |
| 8 | 1423 | 1448 | +2% (in noise) | 0.697 -> 0.709 |
| 16| 1282 | 1344 | +5% (in noise) | 0.786 -> 0.823 |
Decode rate cumulative 157 -> 194 MB/s (+24%). Wall +2-9% (decode is OVERLAPPED so wall gain < decode gain). The word-copy (kill scalar tail, 98.6% of back-refs are len<16) gave the clearest single decode bump (184->194). FASTLOOP + table swap were ~neutral on decode but kept (correct). 
TRAJECTORY POSITIVE (vs the earlier 5-TIE era) — the whole-system LAYERED method works. Still ~0.71-0.82x rapidgzip.
NEXT LEVER (system reasoning): decode-rate gains are now overlapped/diminishing on the wall; leverage shifts to (a) consumer in-order serial path / per-chunk latency (the wall binding), (b) the SYSTEMIC ~1.6x-across-all-components gap (likely memory: u16 2x width / allocator / DRAM-bound per old TMA). Keep layering toward those.

## 2026-05-31 — ADVISOR: pivot to lever A (eager post-process during stall); memory/u16 DISPROVEN
Advisor (code-grounded) verdict for the next lever now that decode gains are overlapped:
- REJECT B (memory/u16 2x width): DISPROVEN — rapidgzip uses identical u16 MarkerVector (DecodedData.hpp:23), so u16 width is NOT a divergence; the diffuse 1.6x-across-components is CONSUMER-STARVATION (workers idle-gated by the consumer's serial submit), not DRAM. (This also explains why decode micro-opts tied: workers already idle-gated.)
- LEVER A (confirmed in vendor source + my own memory project_wall_is_consumer_critical_path): port rapidgzip's queuePrefetchedChunkPostProcessing (GzipChunkFetcher.hpp:521-581). gzippy submits post-processing ONE-chunk-at-a-time in-order (chunk_fetcher.rs:1451) and leaves ready prefetched SUCCESSOR chunks idle during the consumer stall (the wait at chunk_fetcher.rs:1338). Eagerly apply_window/post-process ready successors DURING the stall → convert stall into useful work → shorten the next blocking chunk's latency.
- CHEAP CONFIRM: ~20-line gated probe — during wait_replaced_markers, submit_post_process for ready prefetched successors (published windows). N=0 (today) vs unbounded; frozen interleaved measure.sh. If wall moves → A confirmed AND it's the impl.
- CAVEAT: a PRIOR consumer-postprocess-pump TIED because it enqueued 0 tasks (cache empty during stall). The probe MUST report how many ready successors it finds — if ~0, the real lever is PREFETCH DEPTH, not eagerness.
- DISPROVE memory cheaply (no perf counters): thread-count per-worker-rate sweep — flat per-worker decode MB/s across T2-16 = consumer-bound (A); falling = memory-bound (B).
- Parity REACHABLE: no proven structural cap (u16 + 31% speculation fraction both equal to vendor).

## 2026-05-31 — early-window-publish (byte-identical) + T-sweep: the CHAIN INVARIANT is the root structural lever
T-sweep (per-worker window-absent decode rate, gz_wordcopy): aggregate 215(T2) -> 210(T4) -> 190(T8) -> 163(T16) MB/s, body_ms 746->986. Decode rate FALLS with threads (-12% T2->T8, -24% T16) + body_ms +32% = a memory-CONTENTION/scaling component (refines advisor's "pure consumer-bound"; the u16-WIDTH sub-hyp stays dead = vendor same, but a general bandwidth/contention bound survives). Single-run, suggestive. ⇒ gap is MULTI-FACTOR: serial window chain + contention.
Early-window-publish (chunk_fetcher.rs run_decode_task): worker publishes its last_32kib_window_vec at chunk_end_bit immediately after decode, GATED on max_acceptable_start_bit==encoded_offset_bits (exact-offset chunks only) + provably-clean tail. Byte-identical (routing 25, deflate_block 19, e2e T1-16 vs gunzip). BUT telemetry: published=2/3/1 vs range_speculative=10/17/28 at T4/8/16 — it fires for only 1-3 chunks; the SPECULATIVE marker-bootstrap chunks (the serial-chain bulk) are EXCLUDED because their boundary is a RANGE (max>encoded), unconfirmed until the consumer reaches them.
ROOT STRUCTURAL LEVER (confirmed from a fresh angle): the CHAIN INVARIANT. gzippy demands exact boundary equality (chunk_fetcher.rs:1152/1194); rapidgzip accepts a confirmed range-chain (chunk_N.end==chunk_{N+1}.start at max). Without it, speculative chunks can't early-publish → the serial resolution chain stands. This is THE remaining structural change for parity (+ the contention component). Early-publish kept (correct, layered) though it fires for few chunks today — it banks once the chain invariant lands.

## 2026-05-31 — CHAIN-INVARIANT design: my premise was WRONG; real lever = speculative-window-slot + promote-on-accept (Design B)
Design agent (vendor-source-grounded) correction: gzippy's speculative range [partition_seed, decode_start] is a PREFIX-GAP range, NOT vendor's ALIAS range. Vendor [encoded,max] = stored-block zero-PADDING (non-data) so every offset aliases the same block → range-accept is byte-safe (GzipChunk.hpp:716-722,827-832; Uncompressed.hpp:85). gzippy's [partition_seed, decode_start) bits are REAL DATA decoded by the PREDECESSOR chunk; the chunk's data starts only at decode_start=max. So accepting S<max would DROP [S,max) bytes = corruption. ⇒ gzippy's current `handoff_at_decode_start` guard (S==max, chunk_fetcher.rs:1205) is ALREADY correct + maximally-safe for its range kind; the chain is ALREADY established (chunk_fetcher.rs:1442-1448 + :973 + set_encoded_offset port). "Relax exact-equality to range-accept" = UNSAFE (would corrupt). My/prior-advisor framing was wrong.
REAL LEVER (Design B): speculative dynamic chunks ARE correct data at decode_start; they just can't EARLY-PUBLISH their window (consumer hasn't confirmed decode_start==predecessor-end yet). Fix: worker early-publishes the chunk's CLEAN-TAIL window (predecessor-INDEPENDENT, no markers) into a SPECULATIVE side-slot keyed at decode_origin_end; consumer PROMOTES speculative->real key on accept (identity-key move, bytes unchanged), EVICTS on reject. Breaks the serial resolution chain WITHOUT touching the correct accept guard. Biggest risk: stale speculative window if evict-on-reject isn't atomic -> resolves successor against wrong bytes; catcher = test_single_member_routing_multithread (byte-perfect T4/24MiB). Design A (true alias ranges for stored blocks) is faithful vendor port but ~perf-neutral on silesia (no mid-stream stored blocks); land its range_is_alias tag alongside for fidelity.
