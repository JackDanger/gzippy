# Plan: locate WHY compiled gzippy still loses to compiled rapidgzip (x86 T8)

## The cornered residual (what's already closed, so we don't re-tread)
- T1 = PARITY (IPC 2.05=2.05). The whole gap is T8+ (IPC collapses 2.05→1.55; rapidgzip holds →1.99).
- DECODE is competitive-or-FASTER (window-absent 0.64× cyc/byte, clean 1.04×, ~0.92× instructions). NOT the gap.
- DATA-PLANE (narrowed buffer / data copy / output write) — ported faithfully (a+b+c), measured whole, FALSIFIED: (a) granule = file regression, (b) writev/vmsplice = pipe-only candidate (phantom-pending), (c) warm-retain = inert.
- page-faults = overlapped slack (3 oracles, 0% wall); copies = wall-neutral; consumer-publish = on-path-but-~1.2ms-empty.
THE PARADOX TO CRACK: T8 stalls are TMA-attributed to memory (backend-memory +14pp, DRAM-bandwidth 3× share) — yet every PIECEMEAL memory reduction is slack. Either (i) it's aggregate bandwidth contention removable only holistically (and the data-plane port cut the WRONG bytes / added granule cost), or (ii) it's a coordination/scheduling chain, or (iii) genuinely irreducible. The plan locates which.

## The robust tools we have (and will sharpen)
Fulcrum: memlife (per-buffer mem-traffic, cross-tool, validated), TMA 3-bucket (host paranoid=1→restore, cpu_core pin), `vs` cross-tool span subtraction, schedule/flow (worker occupancy, RATE-vs-PLACEMENT). Perturbation oracles (slow-inject / removal / amplification, +positive +freq-neutral controls = the only VERDICT). Static disasm (cross-tool code). clean_bench frozen harness (trustworthy interleaved wall). All on neurotic, NO local builds, ONE neurotic agent at a time (box-lock + RAM discipline).

## PHASE 0 (pending, independent) — resolve the (b) candidate gain
Build (b)-ONLY (vmsplice/writev over the EXISTING monolithic buffers, NO granule/in-place) + a writev ON/OFF knob. N≥21 interleaved, frozen, PIPE+file, T8/T16. SHIP iff: pipe delta >2σ paired AND vmsplice-counter>0 pipe/=0 file AND file-neutral AND the in-binary ON/OFF control reproduces the pipe delta (the perturbation that resolves the N=9-vs-N=41 sign-flip). ON/OFF reproduces ⇒ real, ship; flat ⇒ phantom, drop. (This is the only confirmed-candidate gain; gate it before the residual hunt or in parallel-sequence on neurotic.)

## PHASE 1 — LOCATE the residual stall region (cross-tool, per-region, NON-smeared)
Q: where do the T8 excess STALL-cycles actually accumulate? (Run-level TMA only exists; per-region was SMEARED at T>1 — never cleanly done.)
- Sharpen Fulcrum: per-region TMA join keyed per-TID-by-timestamp (the fulcrum-total-spec S3 fix), so each region's stall-cycles are un-smeared at T8.
- Measure per-region stall-cycles gzippy vs rapidgzip at T8 + `fulcrum vs` span subtraction: which region does gzippy spend MORE stall-wall in than rapidgzip, EXCLUDING decode (competitive) and data-plane (falsified)?
- OUTCOME: "the residual 1.18× concentrates in region R" (marker-resolve / window-map / consumer-wait / scheduling-gaps / inter-worker), OR "it's uniformly diffuse across all regions" (→ supports the bandwidth-contention hypothesis, Phase 2).

## PHASE 2 — the TOTAL memory-traffic + bandwidth-contention root (the paradox)
Q: does gzippy move MORE TOTAL DRAM bytes than rapidgzip at T8, and is the gap bandwidth-contention?
- Extend memlife to TOTAL DRAM-bytes-moved per MB-decoded, gzippy vs rapidgzip (sum components + uncounted, cross-checked vs perf DRAM counters / IMC if reachable). Is gzippy's total materially higher? By how much?
- THREAD-COUNT SWEEP: gap at T1(parity)/T2/T4/T8/T16 on the frozen harness. Does the gap ONSET track memory-bandwidth saturation (appears as N exceeds BW headroom)? A gap that's ~0 at T1-T2 and grows with N is the bandwidth-contention signature.
- BANDWIDTH AMPLIFICATION perturbation: pin a memory-bandwidth-hog co-runner on spare cores → does gzippy's T8 wall respond SUPER-linearly (gating test: bandwidth IS on the critical path) vs rapidgzip's? Positive control: confirm the hog actually cuts available BW. freq-neutral.
- OUTCOME: gzippy moves N× more total DRAM bytes AND the wall is BW-gated ⇒ the lever is reducing TOTAL traffic (and we now know WHICH bytes from memlife — the data-plane port cut narrowed but added granule; this says what the RIGHT cut is). OR: total traffic ≈ parity / wall not BW-gated ⇒ it's NOT bandwidth, pivot to Phase 3.

## PHASE 3 — scheduling / worker-occupancy (the coordination chain)
Q: are gzippy's workers idle-waiting at T8 in a pattern rapidgzip's aren't?
- Fulcrum schedule/flow at T8: per-worker idle/slack timeline, load-imbalance, the consumer↔frontier serialization, gzippy vs rapidgzip. Is there a scheduling/handoff gap gzippy has (e.g. the in-order consumer stalls the pool, or work-stealing absent) that rapidgzip's design avoids?
- Cross-check vs the known: consumer waits on frontier decode (RATE-100%), frontier already 94% in-flight. If workers are idle-waiting WHILE the consumer waits, that's a schedulable inefficiency; if workers are saturated, it's rate/bandwidth (Phase 2).
- OUTCOME: a scheduling gap (e.g. missing work-stealing / a serialization in the handoff) — faithful-port territory (compare vendor's ThreadPool/FetchingStrategy) — OR confirmed-saturated (→ Phase 2 bandwidth).

## PHASE 4 — causal verdict + lever (or EARNED STOP)
- Whatever Phase 1-3 localizes, run the decisive PERTURBATION (slow-inject / removal / amplification, +positive +freq-neutral controls) to confirm it GATES the wall (not slack). slope≠ceiling: removal-oracle bounds the win.
- Located + gates + faithful-or-authorized fix ⇒ build, measure WHOLE (interleaved N≥21, sha, T-sweep, file+pipe), layer.
- All phases diffuse/slack ⇒ EARNED STOP with the gap fully characterized as irreducible bandwidth/coordination tax (gzippy structurally competitive; x86-only; arm64 already fastest), recorded as a falsification not fatigue.

## Discipline (every step)
Advisor-gate each phase's design + each verdict BEFORE acting. Numbers from the FULLEST Fulcrum harness, never hand-scripts. Perturbation = the verdict, attribution = hypothesis. Validate every instrument (closure/positive-control) FIRST. Frozen, interleaved N≥9 (N≥21 for ~1σ adjudications), sha-verified, RUN_TRUSTWORTHY. ONE neurotic agent at a time; no local builds (RAM). Commit findings as we go; survived-disproof only.
