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
