## BYTE-FLOW — marker = 32% of bytes (structural, NOT 1%); but it's RG-SHARED, so lever-1 = the finalize DELTA only [2026-06-12]
silesia T8 native (DUAL-SHA, 3x stable): marker_bytes 68.17M = 32.2%, clean 143.79M = 67.8%
(accounting complete). 17/18 chunks window-absent (13 flip + 4 finished-no-flip + 1 seeded) — the
"~1% dribble" claim REFUTED; marker is structurally large on silesia T8 because parallel SPECULATION
forces window-absent decode. This mechanically explains NORING 39.5% (ring-write ~31% wall ∝ 32.2%
bytes + apply_window + finalize). BUT the byte-flow worker's lever conclusion ("remove the ring
write = u8-direct that never stores SegmentedU16") REPEATS THE FANTASY the reconciliation advisor
already killed: window-absent markers are UNRESOLVED backrefs that MUST be stored u16 until
apply_window — rapidgzip does the SAME (dataWithMarkers u16, speculates too, ~34.5% marker fraction;
calibration: rg marker_emit 1,743M ≈ gz 2,234M). So the ring-STORE of 32% bytes is RG-SHARED, NOT a
gz-vs-rg lever; NORING's 94ms is a fantasy-vs-rg ceiling. THE DISCIPLINE THAT CATCHES THIS (banked):
a lever = where gz EXCEEDS rg (the calibration insn DELTA), never an absolute removal-oracle that
also removes rg-shared work. THE REAL gz-vs-rg DELTAS (calibration): finalize gz 924M vs rg 174M
(5x — the lever), segmented_ring (gz SegmentedU16 push_slice/extend vs rg FLAT dataWithMarkers — a
real but entangled delta), marker_emit +491M (small), inner-loop frontend/bad-spec (TMA — the big
one). CORRECTED LEVER-1: port rg's 5x-lighter cleanUnmarkedData/finalize — ceiling = the gz-rg
finalize DELTA ≈ 6-7% wall (NOFINALIZE 8.4% x (924-174)/924), ~17ms solvency / ~30ms neurotic, NOT
the NORING 39.5%. Cheap, faithful, wall-validated. A SECOND candidate: flatten SegmentedU16 -> a
flat u16 buffer like rg's dataWithMarkers (kills the segmentation/push_slice overhead WITHOUT
changing width — markers stay u16) — the 24%-of-insn-excess segmented_ring delta; entangled, defer.
STRATEGIC (unchanged): native parity is capped by the inner-loop frontend/bad-spec (engine-W,
asm-bounded 0.667x, HARD); these marker/finalize levers are second-order; drop-isal UNSOUND on Intel.
DISPATCHED: the finalize op-reduction port (lever-1). USER FORK SURFACED: the parity-determining
inner-loop asm bet (hard, possibly capped) vs accepting isal as the x86 ship path.
## RECONCILIATION (Opus heterogeneous) — caught 2 supervisor mechanism errors; u8-width port DEAD; finalize-op-reduction is the cheap win [2026-06-12]
The TMA-vs-NORING contradiction resolved against the CODE. My reconciliation was directionally right
(3 frontend/core levers, consistent on not-memory) but WRONG on 2 mechanisms: (a) push_slice +
clean_unmarked_data are memcpy/VOLUME-bound (not branch-bound) — so "memory<0.5% => width
irrelevant" was unsound; (b) YET the u8-port still captures only ~20ms, for a BETTER reason: the
marker ring is IRREDUCIBLY u16 (MARKER_BASE=32768; markers ARE u16 values [32768,65535] — cannot be
stored u8; rg's dataWithMarkers is u16 TOO) AND the clean bulk is ALREADY u8-direct
(decode_clean_into_contig -> SegmentedU8). => THERE IS NO WIDTH KNOB. **The u8-width SegmentedU8
rewrite is a DEAD LEVER** (= the same 20ms NOFINALIZE region, which is just clean-data-mis-routed-
through-u16). NORING's 94ms is a FANTASY ceiling (markers MUST be stored u16; rg pays it — rg
marker_emit 1,743M ~= gz 2,234M). The achievable faithful subset = FINALIZE OP-REDUCTION (rg
finalize 174M vs gz 924M = 5x lighter), wall ~20ms frozen / ~60ms neurotic.
GOVERNING-MEMORY CORRECTION (surface to user): the u8-clause ("the ring should be u8; keeping it
u16 is the SHORTCUT", project_faithful_unified_decoder) rested on a FALSE premise — vendor ALSO
uses u16 for markers, and gzippy's clean bulk is ALREADY u8-direct. The long-standing "u8 rewrite
owed" mandate is effectively SATISFIED/MOOT; the u16 marker ring is CORRECT, not a shortcut.
RANKED LEVERS: #1 (NEXT) FINALIZE OP-REDUCTION — port rg's 5x-lighter cleanUnmarkedData/finalize
chain (HIGH feasibility, faithful, wall-validated ceiling ~20-60ms), gated by the marker-byte-flow
measurement + a pre-registered falsifier (re-confirm NOFINALIZE; reject if production wall recovers
< spread AND name the residual sub-op). #2 (BIG BET, parity-determining) INNER-LOOP frontend/
bad-spec = engine-W (DOMINANT: 51% of T1 wall, ~36ms symbol-rate gap to ISA-L) but asm-bounded
0.667x — the TRUE native-parity ceiling, maybe unreachable without deeper-than-rung-c asm. DEAD:
u8-width. OWED PRE-FUNDING MEASUREMENT: count bytes through push_slice (marker) vs
decode_clean_into_contig (contig) on solvency T8 — resolves the "marker ~1% dribble (banked) vs
NORING 39.5%" contradiction; decides if lever-1 is a cheap finalize port or a big marker restructure.
STRATEGIC: these levers do NOT reach native parity; the inner-loop is the ceiling; "drop isal"
stays UNSOUND on Intel (isal-recovers confirmed). DISPATCHED: the marker-byte-flow measurement.
## NOFINALIZE/NORING REMOVAL-ORACLE — ring+finalize chain is 94ms WALL; but CONTRADICTS TMA on the u8-port verdict — RECONCILE [2026-06-12]
probe/nofinalize-oracle (pushed): frozen solvency silesia T8, 3xN=9 +/-1ms. NOFINALIZE (skip
clean_unmarked_data's u16->u8 narrow walk) = 20ms (8.4%); NORING (push_slice no-op => ring + ALL
downstream finalize/apply/drain vanish) = 94ms (39.5%); ring-write isolated ~74ms. Neurotic Intel
(unfrozen): NOFINALIZE ~60ms, NORING ~132ms. Both SUBSTANTIAL. Worker concluded "u8-direct port
GO-worthy, ~40-70ms."
**CONTRADICTION (do NOT paper over):** TMA said memory_bound<0.5% => cache/BW dead => u8-direct
NO-GO; this oracle says removing ring+finalize buys 94ms => GO. SUPERVISOR RECONCILIATION HYPOTHESIS
(to be advisor-gated): they are CONSISTENT on the wall being FRONTEND/CORE-bound (not memory) — the
94ms is frontend/core INSTRUCTIONS in the ring+finalize+apply chain, not memory BW. They DIFFER on
what the u8-WIDTH port captures: NORING removes the OPERATIONS entirely; a u8-direct port only
HALVES WIDTH (same elements/iterations/branches, narrower loads) => captures ~the NOFINALIZE 20ms
(the narrow it deletes) NOT the 94ms; width-halving relieves only the <0.5% memory part. The 94ms is
reducible by FEWER OPERATIONS — port rg's lighter post-decode chain (calibration: rg finalize 174M
vs gz 924M insns = 5x lighter), a DIFFERENT change than width-halving. So likely THREE distinct
levers: (1) u8-narrow elimination ~20ms (the u8-direct port's real ceiling, modest); (2) finalize/
ring OPERATION-reduction ~74ms (port rg's 5x-lighter chain — the big post-decode lever); (3) inner
HUFFMAN DECODE loop frontend/bad-spec (TMA: native model 19.5% vs rg 10.8% frontend — a separate
region = port rg's inner-loop structure). The worker's "u8 port buys 40-70ms" likely OVERSTATES
(conflates NORING-removes-ops with port-halves-width). ALSO noted: gz-PURE 237ms vs rg 177ms on
solvency T8 = native 0.74x rg (pure-Rust behind rg on Zen2; the prior "gz ahead" was the ISA-L path).
DISPATCHED: reconciliation advisor (Opus, heterogeneous) — confirm/refute the 3-lever split, the
real u8-direct ceiling, and RANK the levers (wall ceiling x feasibility) before any work-stretch.
NOFINALIZE/NORING are CEILING oracles (garbage output, removal_oracle.rs, production-safe env-off).
## TMA VERDICT — the wall lever is FRONTEND-BOUND + BAD-SPEC in the INNER HUFFMAN LOOP (not BW, not instructions) [2026-06-12]
fulcrum cycles built (TMA-CLOSURE-OR-NO-BREAKDOWN invariant, 348 selftests, pushed fulcrum
2678d4c/bcae041) — Rule-1 tool for cycle/stall questions now exists. THE MEASUREMENT (Intel
i7-13700T frozen + AMD Zen2 cross-check, fractions not absolutes = freq/spread-robust):
MEMORY-BOUND < 0.5% EVERYWHERE => the cache/BW hypothesis is DEAD => the u8-direct SegmentedU8
port is NO-GO-FOR-WALL, CONFIRMED 2nd time (after the falsifier; the NOFINALIZE oracle in flight
corroborates). THE NATIVE-vs-isal/rg GAP IS: FRONTEND-BOUND (I-cache/fetch stalls) + BAD-SPEC
(branch mispredict). model T8: native frontend 19.5% vs isal 10.8% (+8.7pp!) / rg 17.7%; bad-spec
native 28.7% vs isal 23.8% (+5.0pp). silesia T8 smaller but same direction. T1->T8 backend does
NOT jump + memory flat (0.33->0.40%) => NOT BW contention, it's the OOO engine (branches+fetch).
THE LOCATED LEVER: the INNER HUFFMAN DECODE LOOP's branch/frontend structure — native's loop is
branch-rich + I-cache-dense in a way that saturates Raptor Lake's deep OOO frontend and mispredicts
more than ISA-L/rg; rg achieves 2x less frontend-bound on model (10.8 vs 19.5). The charter EXPLICITLY
opened the inner loop for full reimplementation (branchless, multi-literal, BMI2, I-cache layout).
AMD REVERSAL EXPLAINED: native is FASTER than isal on Zen2 (0.239 vs 0.248; simpler frontend, less
branch-pressure-sensitive) — the Intel gap is Raptor-Lake-deep-OOO punishing native's branch-rich
loop = ARCHITECTURE-SPECIFIC MAGNITUDE. "isal recovers on Intel" CONFIRMED at N=15 (model T8 83ms >
30.5ms spread, 0.839x — phantom-killer passed) => "drop isal, ship native" stays UNSOUND on Intel
until native's inner loop closes the frontend/bad-spec gap. NEXT (deterministic): localize WITHIN
the inner loop — perf branch-miss + frontend-stall annotation native-vs-rg (Intel model T8) + a
vendor two-column inner-loop STRUCTURE map (rg deflate.hpp multi-cached loop vs gz asm_kernel
run_contig / marker_inflate) => ranked structural levers (each gated + falsifier-pre-registered
before building); THEN heterogeneous-advisor the inner-loop work-stretch.
## SYNTHESIS (Opus, pivotal RE-AIM) — the wall is CLEAN-DECODE-KERNEL IPC, not instructions/marker [2026-06-12]
Four investigations + the FROZEN removal-oracle reconcile: instruction-count regions != cycle-cost
regions. WALL LOCALIZATION (banked removal-oracle-ceilings, frozen): silesia T1 NODECODE = 642ms =
50.9% of wall; NOSTORE = only 94ms = 7.4%. => the CLEAN DECODE KERNEL (asm/Huffman) is HALF the
single-thread wall — low instruction-count, high cycles/insn (stall/latency-bound). Marker emit is
the biggest INSTRUCTION region but wall-SLACK. CORRECTION: the insn-calib label "kernel 29%
WALL-NEUTRAL" is true ONLY of the SPIN sub-component; the DECODE work is 51% of wall — do NOT write
off the kernel. Q1 RANK: T1 wall = clean-decode/kernel IPC (DOMINANT; 3 pillars: removal-oracle 51%,
engine-W asm-bounded 0.667x, Intel insn-parity-yet-isal-wall-wins => ISA-L higher IPC; native-T1-
no-ISAL is a SUBSET — T1-isal WINS 1.131 on the same bytes, isolating cause to engine not pipeline).
cache/BW WEAK at T1 (NOSTORE only 7.4%). T8 = kernel IPC + scheduling/placement (T4 81% eff; ~40%
T8 prefetch/HoL stalls) + cache/BW (plausible, UNMEASURED at multicore) — REFUSE to over-rank T8
from T1 data (the cross-regime extrapolation that's burned us). Q2 u8-DIRECT PORT (SegmentedU16->U8,
kill the finalize u16->u8 narrow): NO-GO as the wall lever now — it's the 64%-INSTRUCTION-share we
just proved doesn't predict wall, and the u8-falsifier never touched clean_unmarked_data's narrow
(so finalize+ring is UNKNOWN-on-wall, not dead). CHEAP TEST OWED FIRST (Rule 3): GZIPPY_ORACLE_
NOFINALIZE + ring-removal arms (removal_oracle.rs pattern) size the finalize+ring WALL ceiling in
~a day vs the multi-week rewrite. GO on rewrite only if the oracle returns real wall; else
faithfulness-mandate-only, never "the gap-closer," not ahead of kernel-IPC work. Q3 NEXT MEASUREMENT:
TMA top-down stall breakdown (discriminates memory-BW (a) from core-IPC (b)) — build as `fulcrum
cycles` with a topdown closure invariant (retiring+badspec+fe+be==slots), on silesia T8 + model T8
NATIVE-vs-isal-vs-rg, Intel(neurotic) PRIMARY + solvency cross-check + silesia T1 cross-check.
Q4 PHANTOM-KILLER: report TMA FRACTIONS (intensive, freq-cap & spread robust) NOT capped-wall
absolutes; verify dominant bucket is freq-invariant (capped vs turbo on one cell); re-measure
model-T8 isal-vs-native higher-N — if within spread, RETRACT "isal recovers on Intel". Q5 STRATEGIC:
"drop isal, ship native" is UNSOUND on Intel TODAY — isal beats native at T8 on model(+15%)/
monorepo(+45%) + wins all T1; native deficit is STRUCTURAL IPC near the engine-W 0.667x bound
(maybe unreachable without deeper-than-rung-c asm). Pure-Rust as sole FAITHFUL path = OK correctness
milestone; as sole SHIP-PERF path on x86_64 NOW = a measured regression. Do NOT drop isal until
native matches its OWN isal sibling at T8 on model/monorepo on Intel; sequence asm/IPC first,
re-baseline, revisit. DISPATCHED: (1) fulcrum cycles TMA + the stall profile; (2) NOFINALIZE/ring
removal-oracle.
## FULCRUM insn CALIBRATED — 42% marker REFUTED (real 20% of excess); finalize+ring=64% of INSN excess; but insn != wall [2026-06-12]
Calibrated INSN_CATEGORIES vs real solvency perf captures (313 selftests, pushed fulcrum 9808d16) —
Rule 1 satisfied for instruction attribution. CORRECTIONS: marker_emit is 29.7% of total (NOT 42%),
and the EXCESS over rg is only +491M = 20.3% of the gz-vs-rg instruction gap — rg does NEARLY THE
SAME marker work (rg marker_emit 1,743M vs gz 2,234M); marker is NOT the dominant bucket. DOMINANT
INSTRUCTION EXCESS: finalize 40.0% (finalize_with_deflate -> clean_unmarked_data walks the entire
SegmentedU16 BACKWARD + u16->u8 NARROW + CRC migrate) + segmented_ring 24.3% = 64.3% of the
wall-RELEVANT instruction excess; kernel 29.1% is WALL-NEUTRAL (ignore). FAITHFULNESS DISPUTE
RESOLVED toward the 2nd advisor: SegmentedU16 + the finalize u16->u8 narrow IS the shortcut; the
governing u8-clause target (SegmentedU8 / u8-DIRECT clean, eliminating clean_unmarked_data's
narrowing walk) is what attacks the 64% — and the u8-port-v2 worker changed the WRONG sub-lever
(marker-loop mfast COVERAGE, wall-neutral) NOT SegmentedU16. So the REAL faithful-u8 port (ring/
drain u8-direct) is STILL OWED and now precisely targeted. **CRITICAL CAVEAT**: these are
INSTRUCTION shares, and we JUST PROVED instructions != wall on this workload (the marker loop is
wall-neutral despite its instructions; kernel 29% is wall-neutral too). So "finalize+ring = 64% of
INSN excess" does NOT establish it's a WALL lever — it must be removal-oracle-perturbed. The wall is
CYCLES/STALLS (Intel: gap is IPC; u8-falsifier: marker excess is stalls). NEXT MEASUREMENT (Rule 1,
build-tools): a CYCLE/STALL profile (perf cache-misses + stalled-cycles-frontend/backend; a fulcrum
`cycles` analog) to find WHERE the stalls are (SegmentedU16 cache fragmentation? asm-kernel IPC?
memory bandwidth?) — NOT more instruction work. SYNTHESIS DISPATCHED.
## u8 PORT FALSIFIER = FALSIFIED — marker phase is NOT the T8 wall (stalls, not instructions); instruction framing REFUTED [2026-06-12]
Built the faithful u8 single-loop (engine/u8-faithful-v2, pushed, NO MERGE, default-OFF
GZIPPY_MFAST_REENTRY): byte-exact 36/36 sha grid both builds on solvency real-AVX2; marker coverage
55.8%->98.8% (converges to vendor's single readInternal loop); T4 diff_ratio guard PASSES (the prior
1.637 regression did NOT reproduce — kept default-OFF to be safe). THE NEVER-RAN FALSIFIER, now run
on a REAL removal-oracle (not the rule-3-forbidden slope): FALSIFIED on the instruction bar. perf
stat silesia T8: only -21M (-0.32%) total, ~-0.75% of the ~2.76B marker phase; marker phase stays
~2.74B >> 1.9B target. MECHANISM works (44%->1% careful-loop coverage) but the careful-loop excess
is ~32% CYCLES (STALLS), only ~7 insns/converted-event => eliminating it is instruction-neutral.
WALL A/B (interleaved N=11 frozen): ALL TIE (silesia isal 256.5/256.5, native 247.3/247.3; model
isal 121/120, native 135/135). VERDICT: the faithful single-loop is NECESSARY-BUT-INSUFFICIENT —
exactly the off-track read's prediction, now SETTLED by a genuine removal-oracle. The 2nd advisor's
"pass-elimination beats the slope estimate" ALSO resolved: the elimination is byte-exact but
WALL-NEUTRAL. THE GOVERNING "wall = instruction delta" RULE IS DECISIVELY CONTRADICTED on this
workload (2nd confirmation: IPC anomaly was the hint, this is the proof) — perturbing ~42% of
instructions moves the wall 0. PREMISE CORRECTIONS (worker, read-the-code): (a) gzippy marker arch
is ALREADY single-ring (window16 + same-backing ring8 view) + inline in-buffer resolve + in-place
flip — the "SegmentedU16 is the shortcut / flat-buffer is the faithful target" view (2nd advisor via
rg-marker-completing CLAIM B) is DISPUTED; SegmentedU16 = the faithful per-read drain sink (vendor
dataWithMarkers). RECONCILE in synthesis. (b) the 69.7MB marker denominator is SILESIA not model
(model = 1.88MB marker on this box) — the "model-isal" baseline was silesia mislabeled. ROBUST RE-AIM
(survives the faithfulness dispute): the native T8 wall lever is NOT marker symbol rate. It is
STALLS/IPC (cycles), SCHEDULING/PLACEMENT, NATIVE-T1 (no ISA-L), SMALL-FILE overhead, memory
bandwidth — corroborating Intel (gap is IPC not instructions) + engine-W asm-bounded 0.667x.
DISPOSITION: keep u8-faithful-v2 on branch as the measured artifact; do NOT merge (wall-neutral,
default-off = Rule-2 surface for no gain; faithful-arch status is a synthesis question). NEXT:
heterogeneous-advisor SYNTHESIS after the insn-calibration thread lands — what native parity
ACTUALLY requires now that the instruction framing is dead.
## INTEL RE-BASELINE (neurotic, ship-target substrate) — native far from bar; isal RECOVERS, reversing Zen2 [2026-06-12]
24 cells, sha-verified, interleaved best-of-N, native rg ELF (41baa20f, <10ms), built @836e4897.
Intel masks (P-cores 0,2,4,6,8,10,12,14). SHIP-TARGET (native) SCORECARD: PASSES the 0.99 bar ONLY
on storedheavy T8 (1.016, trivial stored copy); EVERY real-content cell FAILS — closest bignasa T8
0.928, silesia 0.84-0.87 all-T, model 0.68-0.77, monorepo T8 0.474 (small-file overhead). The
ENGINE GAP IS THE SAME STRUCTURAL SHAPE AS ZEN2 — NOT a Zen2 artifact, NOT closer on Intel: the
pure-Rust symbol-rate deficit vs rg is real and substrate-independent in SHAPE. HEADLINE REVERSAL:
isal RECOVERS on Intel — beats native at T8 on model (0.474 isal vs 0.558 native, +15%) and
monorepo (+45%), reversing Zen2's "native>isal"; confirms that ordering was a Zen2 PEXT-microcode
penalty on ISA-L, NOT engine truth. On Intel, uncrippled ISA-L makes isal competitive-to-BETTER at
T8 on chunk-heavy real content; native still wins SMT-bound/large (silesia, bignasa) + stored.
T1-isal-wins-all HOLDS both boxes (IsalSingleShot beats rg: silesia 1.146, model 1.075).
RECONCILIATION (critical for strategy): native clean engine is at INSTRUCTION parity with ISA-L
(9.97 vs 10.5 insn/byte, solvency) YET isal wall-BEATS native on Intel at T8 => the gap is
IPC/cycle (microarch), NOT instruction count — Intel ISA-L retires the same instructions at higher
IPC than the pure-Rust asm kernel. So "engine-W done at instruction parity" is FALSE for the WALL:
native needs Intel CYCLE parity, a deeper-asm/IPC problem, not just instruction reduction.
STRATEGIC TENSION (for the user, not yet revisiting): "drop isal, ship native" gives up real Intel
wall on model/monorepo where isal beats native; native-parity is HARDER than the plan assumed.
HIGHEST-VALUE native targets: bignasa T8 (0.928, closest real), the silesia 0.84-0.87 cluster (pure
symbol-rate, same all-T), monorepo (small-file overhead = SEPARATE lever). CAVEAT: no_turbo capped
walls to 1.4GHz base => sub-0.4s cells have >15% spread (min-of-N robust, absolutes low-confidence).
SYNTHESIS PENDING: combine with the in-flight u8-port falsifier + insn calibration, then a
heterogeneous-advisor read of what native-parity actually requires (likely engine IPC + per-cell
levers, not the marker port alone).
## SECOND ADVISOR + USER DIRECTIVE — REDIRECT PARTIALLY OVERTURNED; BUILD the faithful port AS its own removal-oracle [2026-06-12]
Heterogeneous 2nd-opinion (Opus) PARTIAL-overturned the off-track read. HOLDS: 42% split is
UNCALIBRATED (don't quote as decisive), native-T1 bootstrap ABSENT (port doesn't fix T1 — that's
clean-contig asm + 129ms uncharted), engine necessary-not-sufficient. OVER-REACHED (3): (1) the
"12-19ms ceiling" is a RULE-3-FORBIDDEN slope extrapolation — the removal oracle was NEVER run; it
is NOT valid evidence the port is small. (2) marginal perturbation STRUCTURALLY CANNOT measure the
ring-elimination the port does (deletes SegmentedU16 + the separate pass + halves width — a cache/
traffic change a slow-inject can't see; rg-marker-completing CLAIM B sizes ~1.7e9 of the excess as
output-fragmentation); the instruction bucket is the better proxy here. (3) spin lives in the
KERNEL bucket not marker => removing wall-neutral spin RAISES marker's wall-share. KEY RESOLUTION:
the faithful flat-buffer single-loop is FUNDED faithful-arch work (governing unified-decoder memory
+ project_engine_rewrite_funded) that must be built REGARDLESS of wall verdict, AND the port done
right IS the removal-oracle — run the NEVER-RAN pre-registered falsifier (marker insns 2,795M -> ?,
<=1.9B PASS / >2.4B FALSIFIED) + interleaved wall A/B vs NATIVE rg on the engine-W-DOMINANT cells
(model-isal T8 + model-native), NOT silesia (proven slack). Quarantine was the adb05af5
mfast-reentry commit only; the target + criticality STAND.
USER DIRECTIVE (2026-06-12): "It's okay to waste work. We do the hardest best work that might give
value. Explore everything, disciplined, build tools as we go." => BUILD the port (don't gate it
behind measure-first); waste is acceptable; the falsifier tells the truth. EXPLORE in parallel.
DISPATCHED (3 disciplined threads): (A) the faithful u8 single-loop port — recover salvageable
commits, DROP the mfast-reentry regressor, RUN the falsifier + wall A/B (Opus, the hard work);
(B) fulcrum insn category CALIBRATION on solvency (makes 42% deterministic — Rule 1, build-tools);
(C) Intel/neurotic re-baseline native+isal vs native rg (the SHIP-TARGET's real wall, un-penalized
by Zen2 PEXT microcode — explore the real gap). native-T1 (clean-contig + 129ms) + the parked
silesia-isal T4 are SEPARATE owed levers, not this port.
## OFF-TRACK READ (user-requested, READ-ONLY Opus) — DRIFT FOUND: marker lever is NECESSARY-BUT-INSUFFICIENT [2026-06-12]
Cold outside assessment. BAR is tracked honestly (zero corpora pass under the binding bar; T1-isal
20/20 = product-today not goal-state; discipline + quarantine held) — NOT losing sight of the bar.
BUT losing sight of whether the CHOSEN LEVER can reach it. CRUX (the campaign's OWN arithmetic):
marker phase = +1.0-1.5B of the +2.8-3.0B instruction excess (~42%); a PERFECT marker port leaves
~1.55B (kernel +631M, sched +490M, tables +217M, finalize +142M) => still ~1.30x insns => ~0.77x
wall by the governing rule — nowhere near 0.99. Wall arith worse: the criticality probes give marker
only ~12-19ms ceiling against 50-90ms gaps. WORST: the port is MIS-TARGETED for the SHIP build —
bootstrap/marker is STRUCTURALLY ABSENT at native T1 (the flagship cell the rewrite was funded for,
banked INC-0); marker criticality is confirmed mainly on ISAL-T8 (the build being DROPPED). Real
native beneficiary = model/weights-native only (narrow). The u8 port has INDEPENDENT justification
as the faithful-architecture mandate — pursue it AS THAT, not sold as "the #1 gap-closer."
OTHER FINDINGS: (2) the "42%" is an UN-CALIBRATED per-category split the campaign itself flagged
not-yet-trustworthy (fulcrum insn calibration owed) — prioritizing the lever on it violates Rule 1
in spirit. (3) ALL wall numbers are Zen2 (ISA-L PEXT microcode-crippled, penalizes isal); Intel
ship-box (neurotic) un-re-measured. (4) IPC ANOMALY: gz IPC BETTER than rg + kernel "spins-not-
parks" => a chunk of +2.8B is WALL-NEUTRAL SPIN — the instruction=wall framing is contradicted by
its own data and is next-in-line to be refuted (after CPU-gate + wall-conversion already died). (5)
parked failing cells with no plan: silesia-isal T4 0.912, native T1 129ms uncharted. (6) minor:
broken clean-window oracle still compiled; RSS (native +40%/2.7x) not in the bar = unratified scope
narrowing; perturbations still 3-corpus monoculture. RESOLUTION (measure, don't argue — Rule 1):
2nd heterogeneous advisor on the redirect + DETERMINISTIC measurements before funding the port:
(M-solvency) calibrate fulcrum insn categories + classify spin-vs-real-work share; (M-neurotic)
Intel re-baseline native+isal vs native rg + marker criticality on the box that SHIPS. The port is
NOT cancelled — it is reframed (faithful-arch + model/weights-native) and DE-prioritized as "THE
lever" until the data says it closes the bar.
## FULCRUM `insn` HARDENED (code-side trustworthy) — calibration is the remaining SUPERVISOR step [2026-06-12]
The gate's 3 gaps CLOSED + pushed (fulcrum main 5bc1147/2b785e7/82cc031, 228 selftests): GAP2
INSN-EVENT-MISMATCH refusal (denominator/event-mismatch class — the 2.7-insn/byte hallucination —
now caught + firing-tested; alias instructions==inst_retired.any, absent-header => no false
refusal); GAP1 false "safe by construction" docstring corrected + a necessary-not-sufficient test
(single-wrong-bucket CLOSES green yet wrong split — closure protects TOTAL not per-category); GAP3
refusals asserted BY NAME (_raises_named, InvariantViolation). insn is now TRUSTWORTHY CODE-SIDE
for totals AND structurally honest about per-category limits. REMAINING (Rule-1 completion):
the gzippy category patterns (adapters/gzippy.py INSN_CATEGORIES) are PROVISIONAL until calibrated
against real `perf report -F period,symbol` of gzippy-native/isal/rg — a SUPERVISOR run on
solvency (instruction counts) ready via GzippyAdapter.calibration_capture_cmds(). DO THIS before
trusting insn's per-category answer on the marker-phase +2.8B verdict (i.e. before/with the next
u8-port instruction re-measure). Until then: trust insn TOTALS, hand-verify per-category or run
the calibration first.
## FULCRUM `insn` BUILT + GATED — totals TRUSTWORTHY, per-category NOT YET (hardening dispatched) [2026-06-12]
Rule-1 capability landed: `fulcrum insn` (decide/fulcrum/core/insn.py, INSN-CLOSURE-OR-NO-LEDGER,
222 selftests, pushed) — deterministic instruction ledger that REFUSES unless categories sum to
the perf-stat total; the 690M in-tool double-count class is now STRUCTURALLY IMPOSSIBLE
(single-bucket resolve_category + over-count/ambiguous/percentage refusals, all non-vacuous per
the gate). HETEROGENEOUS-ADVISOR GATE (Opus, adversarial) caught Rule-1-critical gaps BEFORE we
trusted it on the +2.8B question: (G1) per-category attribution — the ACTUAL deliverable — has NO
guard; a single-WRONG-bucket symbol passes silently because the TOTAL still closes (the adapter's
"safe by construction" claim is FALSE for single-wrong-match); (G2) NO stat-vs-report EVENT
cross-check => a report sampled on a different event summing within tolerance conserves silently
(the denominator-mismatch class = the old 2.7-insn/byte hallucination); (G3) selftests assert
refusals by exception TYPE only (can rot green). The "conservation assertion" is an algebraic
IDENTITY (can't fire) — the real guards are elsewhere. VERDICT: TRUST insn TOTALS, do NOT trust
its per-category split until calibrated + event-check added. HARDENING DISPATCHED (push-free):
INSN-EVENT-MISMATCH refusal (G2), docstring/comment truth + necessary-not-sufficient test (G1),
refusals-asserted-by-name (G3). CATEGORY CALIBRATION needs real perf data = a SUPERVISOR step
(gzippy-native/isal/rg `perf report -F period,symbol` on solvency+neurotic) before insn's
per-category answer is trusted for the marker-phase verdict. This gate is the user's
"more/heterogeneous advisors" rule paying off — exactly the vacuous-invariant class that slipped
twice before (locate residual, registry subset).
## CLEANUP BASELINE MERGED (ea7c7584) + workspace minimal + neurotic online [2026-06-12]
gzippy easy-mode cleanup MERGED (solvency-verified MERGE-READY: both builds compile, clippy -2,
4/4 sha smoke 028bd002 correct routing, fmt clean): 4 dead-code deletions, stale-comment removals,
39 truly-orphaned plans archived (zero dangling citations). Workspace MINIMAL (4 worktrees/5
branches; recovery patches /tmp/gz-cleanup-patches, 20 files). Submodule gitlink repaired
(vendor/isal-rs/.git was cross-linked to deleted gz-vol worktree -> repointed to this checkout's
module dir; git status rc=0 clean). release/formula-0.7.1 reconciled (reset to origin-authoritative;
2 stale dup commits saved as patch). NEUROTIC ONLINE (ssh -J REDACTED_IP root@REDACTED_IP, key
works, corpora present) = the INTEL wall/cycle box. FULCRUM push freely authorized (main pushed
801736a). IN FLIGHT: fulcrum agentic-surface polish (push-enabled, building fulcrum insn mode),
gzippy source-easy-mode cleanup (env-knob cull + dead-code + comment-truth, branched off the merged
base). NEXT GZIPPY CLEANUP (sequenced post-source-merge to avoid conflicts): docs/naming pass.
USER GUIDANCE 2026-06-12: build/fulcrum portability is first-class — if any of the 3 machines
(macOS/neurotic-Intel/solvency-AMD) fails to build, or fulcrum fails on any arch, make a fix-plan +
escalate to user or an Opus agent.
## WORKSPACE MINIMIZED (user Rule 2) — 71->4 worktrees, 181->5 branches, ~30Gi reclaimed, zero loss [2026-06-12]
Two reversible passes: 67 worktrees retired + 176 local branches removed; FINAL local surface =
ONLY the protected/active set: worktrees {gzippy[main], gzippy-reimplement-isal[reimplement-isa-l],
/tmp/gz-clean[chore/cleanup-easy-mode, verify in flight], /tmp/gz-u8loop[engine/u8-single-loop,
quarantined]}; branches those 4 + release/formula-0.7.1. REVERSIBILITY (nothing destroyed): every
unique commit pushed to origin (39 pushed, 21 already-reachable); 17 dirty-worktree diffs + lowt's
5 untracked files saved to /tmp/gz-cleanup-patches/ (MANIFEST.json) — all dirty content was benign
(bench scripts, Cargo.lock, scratch .rs/.md). Disk 1.3Gi->32Gi free. FLAG (user): release/
formula-0.7.1 DIVERGED from origin (2 local-unique commits, also behind) — NOT force-pushed; needs
`git pull --rebase origin release/formula-0.7.1` then push to preserve those 2 commits. OPERATING
RULES now load-bearing ([[feedback_operating_rules_2026_06_12]]): Rule 2 (clean workspace + truthful
names) just executed — also pushed 2 opaque worktree-agent-<hash> branches under truthful names;
Rule 1 (Fulcrum-for-deterministic-answers) governs all future measurement; advisors reverted Fable
->Opus, used MORE + heterogeneously. Recovery-patch dir /tmp/gz-cleanup-patches is itself transient
surface — deletable once the user confirms nothing's needed from the retired probes.
## u8 PORT QUARANTINED — fails T4 pipeline-speedup guard (1.226->1.637); falsifier never ran [2026-06-12]
Verification of the recovered orphaned branch engine/u8-single-loop (sonnet, solvency, GitHub-key
fetch): DIFF AUDIT clean (asm_kernel.rs + decode_clean_into_contig UNTOUCHED — parity path safe);
build+flavor OK; but FULL SUITE FAILS both builds on diff_ratio_parallel_single_member_speedup —
T4/T1 ratio regresses 1.226 (base 72bb692d) -> 1.637 isal / 1.689 native, the mfast RE-ENTRY
commit (adb05af5) makes T4 ~2.1x slower (per-chunk latency up; re-entering 'mfast after the wrap
adds work that costs at low-chunk-count T4). NO correctness bug (944/952 pass, no sha mismatch).
The pre-registered FALSIFIER (marker insns 2,795M->?) NEVER RAN (stop-on-first-failure). VERDICT:
branch QUARANTINED, NOT merged (failing guard test = automatic no-merge). The THREE commits split:
51788caf (delete dead trailing_clean scan) + fe36b8c2 (thread-local test counter) are likely
SALVAGEABLE clean; adb05af5 (re-entry) is the regressor — kill-switch GZIPPY_NO_MFAST_REENTRY=1
isolates it. DEFERRED (not now — user redirected to cleanup; Fable porting model is DOWN anyway):
re-attempt the u8 port WITHOUT the re-entry mis-step, or with re-entry behind the kill-switch
OFF-by-default, then run the falsifier. The marker-phase target (+1.0-1.5B, 42%) and its
criticality verdict STAND — only this particular implementation regressed. MODEL: advisors revert
to OPUS (Fable inaccessible; user-confirmed 2026-06-12). CLEANUP CAMPAIGN LAUNCHED (user-directed,
easy-mode for a smaller driving model): Opus workers on (a) Fulcrum trustworthiness audit
(invariant-firing proofs, vacuous-test fixes, dead/false-doc deletion; cleanup/trustworthy, no
push) and (b) gzippy codebase cleanup (dead-code from compiler warnings, env-knob keep/delete
audit vs banked verdicts, stale-comment deletion, plans/ graveyard -> plans/archive/,
implicit->explicit; chore/cleanup-easy-mode, byte-exact).
## u8 PORT WORKER DIED MID-VERIFY (Fable model access error) — work RECOVERED + re-verifying on solvency [2026-06-12]
The u8 single-loop port worker (Fable) hit a model-access error (claude-fable-5[1m] inaccessible)
after ~2h/131 tool-calls and returned only the error — but it had COMMITTED 3 coherent commits to
engine/u8-single-loop (now PUSHED to origin): inlined backref resolution into the marker read
loop, single-loop fastloop re-entry (careful loop yields back to mfast after ring wrap), deleted a
dead trailing-clean scan. ~597+/509- in marker_inflate.rs + gzip_chunk.rs; clean tree. NO
verification report exists => UNTRUSTED until re-verified (the rewrite is byte-exactness-critical:
the P0 word-copy-overshoot class lives on exactly this surface). RECOVERY: branch pushed; a
verification worker (sonnet — Fable is the access failure) FETCHES it on solvency (GitHub key now
installed) and runs the full gauntlet + the pre-registered falsifier the dead worker never
reported (marker-phase insns 2,795M -> <=1.9B same 69.7MB denominator; 48-cell sha grid;
asm/clean-parity-path-untouched audit; wall A/B vs 72bb692d). MODEL NOTE: Fable inaccessible this
session (user switched session default to Opus 4.8); the standing "Fable advisors" directive
(feedback_advisor_consult, 2026-06-12) is temporarily UN-followable — advisors/workers fall back
to Opus/sonnet until Fable returns. "0% CPU on solvency" the user observed = the port was in its
LOCAL Rosetta correctness phase; solvency goes busy now with the verification fetch.
## MARKER EMISSION WALL-CRITICAL (both builds, solvency T8) — u8 SINGLE-LOOP PORT IS GO [2026-06-12]
probe/marker-emit-crit (c459ee54 on solvency). Knob inside BOTH emit sites (mfast :2349 + careful
:2567), loop-gates verified untouched (new marker_emit_spin var, mfast entry unaffected),
OFF-vs-absent + DUAL-SHA clean. Pre-registered, calibrated (Zen-2 ~1.1ns/iter; N=100 ~= +1-1.4x
emit phase). RESULTS (stable rounds, std 3.6ms): native +6.9/+18.8ms at +50/+100 sleep (1.9x/5.2x
spread), spin +20.0 ~= sleep => NOT turbo; isal +2.6/+8.1ms (sleep) / +9.6 (spin). VERDICT:
monotonic + proportional + frequency-neutral-surviving = MARKER EMISSION CRITICAL at T8 both
builds. Anomalies: rounds 8-9 burst-state artifact (excluded, documented); MFAST_PROF silent on
solvency (gate flag, uninvestigated); worker fixed 2 pre-existing clippy errors in
crates/gzippy-inflate in-commit. THE PORT IS GO on all grounds: instruction ledger (+1.0-1.5B,
banked), wall criticality (this probe), vendor shape (Block::read inline resolution,
deflate.hpp), and the governing u8 clause (ONE MarkerRing, in-place u16->u8 flip — the funded
engine-campaign core step). PRE-REGISTERED PORT FALSIFIER (banked, unchanged): marker-phase user
insns 2,795M -> <=1.9B on the 69.7MB denominator (perf-record, solvency); >2.4B = mechanism
FALSIFIED (intra-loop rate, not the pass split). Wall expectation: >=~12ms at solvency T8
(emit-share arithmetic), larger on Intel. DISPATCHED: the u8 single-loop port (Fable worker —
the campaign's funded structural rewrite, full engine-core gauntlet).
## INSTRUCTION-DIFF GATED (ledger REBUILT on-box): MARKER PHASE = #1 TARGET (~42%); CLEAN ENGINES AT PARITY [2026-06-12]
Gate REJECTED the worker's categories (and the supervisor's candidate): measured denominators
decided it. VOLUMES (measured, parity confirmed — placement converged): gz marker 69.7MB / clean
142.3MB ~= rg 67.6 / 144.4. ISA-L CALIBRATED on this box (igzip CLI, same lib): ~10.5 insn/byte —
instruction counts are SUBSTRATE-INDEPENDENT (PEXT microcode = cycles, not counts; the worker's
"ISA-L 2.7/B" was a denominator hallucination — rg's ISA-L covers only ~39MB window-known chunks;
its clean BULK is the custom u8 path inside Block::read at ~12-14.5/B, vendor flip deflate.hpp:1282).
CORRECTED LEDGER (gz-native vs rg, +2,995M): #1 MARKER PHASE +1.0-1.5B (~42%): gz 40.1
insn/marker-byte across THREE passes (read 15.2 + emit_backref 16.9 + segmented 8.0 = 2,795M)
vs rg ~18-26/B in ONE loop — the u16 two-pass split ~doubles the marker rate; #2 kernel +631M
(syscalls EXONERATED: gz 1,030 vs rg 3,244 — gz spins-not-parks; faults +46.6K explain <=20%;
mechanism OPEN); #3 sched +490M; #4 tables +217M; #5 finalize +142M; CLEAN DECODE ~0 — gz asm
9.97/B vs ISA-L ~10.4/B = PARITY-or-ahead: "asm-vs-ISA-L instruction parity" is ALREADY ACHIEVED
(engine-rate arithmetic predicts the isal-native delta -61M EXACTLY); any clean-side asm grind on
instruction grounds is REFUTED (revisit only with Intel cycle data). ANOMALY RESOLVED: the
"unprofilable 1.57x isal" rebuild had DROPPED FEATURES (legacy-serial flavor — the guard caught
it); truth table uncontaminated. RATES BANKED (silesia, insn/output-byte): ISA-L ~10.5, gz-asm
9.97, rg-custom-u8 ~12-14.5, rg-marker ~18-26, gz-marker 40.1.
THE STRUCTURAL FIX = the long-governing u8 clause (ONE MarkerRing, in-place u16->u8 flip, backref
resolution INLINE in the read loop, deleting emit_backref + segmented_markers as separate passes
— vendor Block::read shape): the instruction profile just gave that directive its measured
justification. DISPATCHED: marker-phase criticality perturbation (emit_backref +50% sleep, T8,
pre-registered) per process rules; THEN the single-loop port with the PRE-REGISTERED falsifier:
marker-phase user insns 2,795M -> <=1.9B (<=27/marker-B) = total excess shrinks >=0.8B;
FALSIFIED if single-loop lands >2.4B (rate gap would be intra-loop). Addendum on solvency:
plans/instruction-diff-v1.md.
## SMALL-FILE RECON: NO vendor mechanism to port — clamp + lazy-spawn already faithful; one micro-divergence named [2026-06-12]
Read-only vendor recon (ParallelGzipReader/ThreadPool/BlockFetcher + gzippy startup ledger):
rapidgzip has NO sequential fast path / NO thread clamp for small inputs — its ONLY small-file
mechanism is the chunk-size clamp (PGR.hpp:294-306, 2MB@-P8 => 512KiB chunks => 4 chunks),
ALREADY PORTED verbatim (single_member.rs:75-88); lazy one-at-a-time thread spawn ALREADY PORTED
(ThreadPool.hpp:157-159 == thread_pool.rs:325-349); rg's ChunkFetcher is lazily built on first
read (gzippy builds eagerly in drive() — mild). ONE FAITHFUL MICRO-FIX NAMED: gzippy calls
core_affinity::get_core_ids() (syscall) EAGERLY in ThreadPool::with_pinning_for_capacity per
drive(); vendor pins lazily inside workerMain — move the query to spawn time (OnceLock global).
Plus ~12 OnceLock env reads + per-iteration Instant::now() pairs with no vendor equivalent —
sub-ms each, visible only at 30-80ms walls. RECON'S CONCLUSION partially CAVEATED by supervisor:
"pure-Rust engine slower is the dominant factor" does NOT explain the ISAL build's markup.xml
0.373 (isal uses ISA-L for clean tails) — the small-file deficit on isal must be fixed-overhead +
per-chunk marker bootstrap at 30-80ms scale; the running instruction-diff profile owns the real
answer. DISPOSITION: no clamp work (nothing to port); the lazy-affinity micro-fix queues behind
the profile verdict; small-file cells fold into whatever the convergence list names.
## SOLVENCY TRUTH TABLE v1 BANKED — first DIRECT instruction counts: BOTH builds retire ~1.55x rg's instructions [2026-06-12]
infra/solvency 9a780784 (plans/solvency-truth-v1.md on the box; canonical 19-file Squishy set
freshly pinned, encoder gzip 1.12 -9, manifest in the doc). N=7/5 frozen interleaved, sha-OK,
strike-5 clean, native rg comparator (5ms). FRAMING (advisor-fixed): (1) T1 isal 20/20 PASS is
the PRODUCT-TODAY scoreboard; the GOAL-STATE (pure-Rust sole path) T1 column is native = 2/20 —
carry BOTH, never quote "T1 solved". (2) Under the binding bar (>=0.99 at EVERY T): ZERO corpora
pass. (3) T8: isal 3/20, native 2/20 (tiny files); silesia 0.720/0.796; markup.xml 0.373 (2MB
file, full 8-thread spin-up — rg small-file mechanism UNVERIFIED, recon dispatched). (4) native >
isal at T8 on Zen 2 (0.299 vs 0.330) = SUBSTRATE-SPECIFIC candidate (ISA-L kernels lean on
PEXT/PDEP, microcoded on Zen 2; isal IPC 1.773 vs native 1.886) — do not generalize.
THE HEADLINE: perf stat silesia T8 — gz-isal 7.709B insn / gz-native 7.611B / rg 4.897B = +2.8B
excess (~13 insn/output-byte), BOTH builds ~1.55x, gz IPC BETTER than rg (spin-loop signature
candidate!). Cross-substrate replication of the i7 1.5-1.7x rate verdict (ratio replicated;
mechanism not yet). BUILD-IDENTICAL COUNTS (7.7 vs 7.6B with the clean engine swapped on ~210MB):
either the engines retire within ~0.5 insn/byte of each other OR isal routing is mislabeled —
STEP-0 DISCRIMINATOR OWED (VOLUME-AUDIT counters + perf stat same invocation, both builds) before
the shared-machinery implication banks. Arithmetic pre-kill: memcpy-excess CANNOT explain 2.8B
(~0.03 insn/byte); live suspects = scalar per-byte loops (marker flip/window substitution),
per-symbol overhead (~50 insn x ~55M symbols), or BUSY-SPIN (IPC signature). CPU silesia T8:
1.56/1.45/1.04s; RSS 320/296/228MB (+40%, marker-ring/u16 suspects).
DISPATCHED: (1) perf-record instruction-diff on solvency — fp call-graphs (NO LBR on Zen 2;
force-frame-pointers rebuild + self-test within spread), sample instructions:u fixed period,
user/kernel split FIRST (RSS excess may be kernel page-faults), symbolized rg rebuilt and
verified-equal to the pinned comparator, interleaved+sha-verified records, accounting closure
~15%, two-build filter (similar-self-count = shared suspect) + spin-symbol tagging,
PRE-REGISTERED: ">=60% of the +2.8B sits in build-shared functions" (falsified if build-divergent
concentrates), output = RANKED CONVERGENCE LIST, hypothesis-generator only — each top entry gets
a perturbation or delete-divergent port before banking. (2) markup.xml vendor recon (read-only:
rg's small-input clamp mechanism, file:line; NO code change before the vendor counterpart is
named).
## VOLUME AUDIT (analytical, PROVISIONAL): volume ~parity => RATE owns the 1.54x — and NOT the marker loop [2026-06-12]
probe/decode-volume (c2ba564: adds the one missing counter ISAL_ENGINE_ORACLE_BYTES + a
VOLUME-AUDIT verbose line; live guest run BLOCKED — the worker's network position lost neurotic;
live verification moves to solvency post-provisioning). ANALYTICAL DISCRIMINATOR from banked
counters: both tools push ~284MB through inflate engines on silesia (73MB marker-mode + ~210MB
ISA-L; the 32KiB-flip architecture is shared; spec-fail discards <5MB; apply_window is
substitution not re-decode) => gz/rg ENGINE-VOLUME ~1.0 => the 1.54x thread-summed excess is
RATE on equal volume. SHARPER: the marker phase is where gz is FASTER (226-269MB/s vs rg custom
~153MB/s, ~0.3s vs ~0.48s) and the pure-Rust clean loop sees only ~1.5MB on the isal build =>
the excess must sit in the ISA-L-path per-chunk/wrapper/pipeline term (DIS-14's T1 residual
0.903x class) — TENSION to resolve: the model FFI-surface probe measured wrapper at ~0.1-2%
(but on MODEL; silesia never got the FFI-surface ledger at the native bar). ALL PROVISIONAL
until the live VOLUME-AUDIT line + a silesia per-phase cycles/byte ledger run — ON SOLVENCY,
where bare-metal perf can count retired instructions per phase directly (the better instrument).
GZIPPY_SLOW_MODE caveat banked: on the isal build it perturbs only the ~1.5MB pure-Rust clean
slice — flat results there say nothing about the ISA-L path. MIGRATION: bench host moving to
solvency root@192.168.7.222 (user-directed; AMD EPYC 7282 Zen 2 bare metal, full perf+IBS;
provisioning in flight; Zen-2 PEXT/PDEP-microcoded caveat for the asm kernel; all banked
absolutes are i7-specific — re-baseline owed; neurotic = legacy, reachable from supervisor shell
but not from some worker network positions — anomaly noted).
## CPU GATE DIES — rg measured natively on 3 binaries: gz pays 1.5-1.7x rg's decode CPU; ENGINE-W IS THE LEVER (quantified) [2026-06-12]
Wall-conversion gate (probe/wall-conversion f9754052 gated; advisor re-measured rg itself, frozen,
17 sha-verified runs, 3 rg binaries AGREE): rg silesia-T8 truth = wall_min 280ms (pin stream),
CPU thread-summed ~1.50-1.53s => /8 ~= 188ms, ThOpt 166-181ms, Eff 75-82%. THE BANKED rg CPU
2105ms (=> 263ms/8) REPRODUCES ON NO BINARY incl. the trace build it was attributed to — VOID,
plus every downstream use: the CPU gate "gz <= rg" DIES for silesia T8, and the "ONE shared
wall-conversion mechanism at T4+" re-aim dies with it (built on the dead gate). LOCATED (wall
delta = instruction delta, per the governing rule): gz decodeBlock ~2.18-2.38s vs rg ~1.5s =
~0.9-1.0s thread-summed EXCESS DECODE CPU; scheduling near-parity (gz 79.5% vs rg 75-82%) =
non-differential. ENGINE-W RE-CONFIRMED as THE silesia/T4+ lever with the convergence metric:
gz ThOpt 272-297ms -> 178ms on pin-silesia T8. BANKED: canonical rg verbose invocation
`rapidgzip -d -f -o <file> -P 8 --verbose` (returns in-wall; the worker's 18-min "hang" was an
invocation error building a seek index). bignasa T8 0.829: DIFFERENT mechanism — no ISA-L
engagement (flip=0, isal_chunks=1), consumer/output 219ms = Wall-Real named term; the
pre-registered Finalize/prepend perturbation aims THERE (chunk-11 finalize=208ms). DOWNGRADED:
tail window-publish waits (193-255ms ffi_outside) — real but capped ~0-20ms differential by
efficiency-parity; hypothesis only. FALSE ALARM RESOLVED: silesia.gz pin f16dd24b is INTACT
(mtime 06-06); the worker used the staged repo fixture (96070b0a, same raw bytes, different
encoder) — its 0.810 session is internally valid but NOT comparable to pin-era ratios (re-run
owed on the pin); STRIKE-5 VIOLATION: the worker demoted the sha-mismatch abort to WARN — rule
hardened: pin mismatch = hard abort, staged copies inherit the pin check, corpus files immutable.
NEXT (ranked): (1) re-derive the gz verbose pair on PIN-silesia T8 (~10min, restores
comparability) then ENGINE-W against the 0.9-1.0s excess (where does gz burn 1.5-1.7x rg's
instructions? per-symbol rate was partially exonerated by MFAST_DISABLE-flat at T4/T8 — that
tested the MARKER loop; the CLEAN/contig+asm path and per-chunk re-decode volume (speculation
overlap/discards, double-decode tax) are the candidates — a decode-VOLUME audit (bytes decoded
total vs 211MB; rg 31.25%-marker double-decode vs ours) is the cheapest discriminator BEFORE any
kernel work); (2) Finalize/prepend perturbation on bignasa vs native bar; (3) process fixes:
RG pinned to b0397fca in scripts (done in rescore branch — merge it), strike-5 hard abort, hunt
the 2105ms artifact origin before trusting any RG_TRACE CPU table.
## N2 TIE-KEEP MERGED (1d008f71) — rate-trap untriggered; ladder dims; disk-full make scare resolved [2026-06-12]
F-w1 verdict per pre-registration: TIE (model-isal T8 median 1.0043, on/off byte-exact 12/12
spot-grid, counters identical, no-regression silesia 1.0043 / bignasa 0.987) => keep-if-byte-exact.
BODY-RATE BANKED: ~48->50MB/s (1.04x) — the SECOND consecutive micro-pass (after N1) that did not
move the marker body rate; rate-trap (>=1.5x) untriggered so the slope ceiling stands, but the
N3-N5 ladder prospects DIM (two TIEs, flat rates) — ladder PAUSED pending the wall-conversion
decomposition (in flight; the dominant unknown owns the next direction). Honest model-isal T8 gap
vs native rg (oracle_c/build-native, labeled): 1.15x. INCIDENT: post-merge `make` failed
StorageFull — local disk 99% from accumulated worktree targets; freed 3.9G (finished workers'
target/ dirs), make green (EXIT=0); the merge was sound. LESSON (process): `make 2>&1 | tail`
swallows make's exit code via the pipe — the push gated on tail's 0, not make's failure; always
capture MAKE_EXIT separately (this bank is the record; the pushed merge was retroactively
verified green). IN FLIGHT: wall-conversion decomposition silesia T8 vs native rg (the campaign's
single dominant unknown).
## NATIVE-RG RE-SCORE GATED SOUND — campaign re-aims: T4+ wall-conversion is THE gap; T1-isal wins real [2026-06-12]
measure/native-rg-rescore (db466ae5) gated. Comparator = native ELF b0397fca (4ms startup,
self-check wired into _parity_guest.sh — wheel regression now mechanically trapped); N=7 frozen,
sha-verified, strike-5 clean. CONVENTION: ratio = rg_native_min/gz_min, >=0.99 PASS. CROSS-CHECK
PASSED: factor-walk's independent ~1.17x silesia-T8 reproduces (0.848 = 1/1.18) on different
absolutes — two instruments agree. SCOREBOARD: PASS only at T1 (silesia-isal 1.131, model-isal
1.108, storedheavy 1.372 isal/1.099 native); EVERY T4/T8/T16 cell LOSES 0.65-0.96. Wheel tax is
DATA-SCALED 42-85ms (58% of rg's storedheavy time; bignasa 85ms > 48ms startup => runtime Python
IO component — wheel invalid at any size; NO fixed-subtraction reconstruction of old numbers is
ever valid). Six comparator-swap flips PASS->LOSS (silesia/storedheavy/storedmix T8 x both
builds) — column is same-session swap, NOT a wheel-era reproduction (gz drifted faster since).
SURVIVES: all same-binary causal results (perturbations, kill-switch A/Bs, CPU ledgers,
exonerations), rg CPU tables (RG_TRACE = native build-trace, verified), CPU gate gz<=rg, all T1
wins, all faithfulness merges (keepIndex/staging/demotion-mechanism). DIES/RE-OPENS: F1
win-at-parity, "silesia T4 drift-limited terminal" (0.816 is not drift; reboot framing moot),
bignasa 0.986, weights 1.03, storedheavy/storedmix T8 WIN headlines, rg-side 92% efficiency
(wheel wall; gz-side 81% stands; re-derive natively — gap widens). RE-AIM: gz decode CPU <= rg
yet wall 15-35% worse at T4+ UNIFORMLY across corpora => ONE shared wall-conversion mechanism
(scheduling/overlap/output), ~50ms on silesia T8 — order of magnitude beyond the old 3pp story.
RANKED: (1) N2 F-w1 finishes (in flight, disposes bootstrap track); (2) silesia-T8 wall-conversion
decomposition vs native rg (Gantt+ledger, both sides native, re-derive efficiency pair natively
same-session; piggyback bignasa N=7 re-run); (3) Finalize/prepend_bytes perturbation (pre-reg,
both corpora, native bar). NO-LOCK: weights/monorepo re-fetch via host (squishy 403 on guest).
PARKED: release-correction PR (user nod). NITS: T16 mask = 15 cores (T16 anomaly under-read until
re-run); silesia-T16 native 325 < isal 364 banked as anomaly only.
## COMPARATOR-INTEGRITY BOMBSHELL: banked rg numbers carried a ~43ms PYTHON-WHEEL STARTUP TAX — silesia T8 "parity" RE-OPENED [2026-06-12]
Ledger-pass gate reconciled the rg 350->274 contradiction by FACTOR WALK (frozen, N=7, sha-OK):
R0 worker-form (native-rg, /dev/null) 273ms == worker's read; +file sink => 326 (SINK +53ms);
+flags TIE; +PATH binary => 376 (BINARY +52ms, of which ~43ms = the pip WHEEL's Python console-
script startup: wheel --version 47-48ms vs native ELF 4-5ms; residual codegen ~9ms). BOX STATE
UNCHANGED (R3 reproduces the banked regime; "1.383GHz power-capped" = TSC base clock misread —
does NOT strengthen the reboot ask; fd_vectored env flag stands alone). THE STING: parity.sh's
PATH rapidgzip IS the wheel => EVERY banked rg number paid ~43ms startup. Against the NATIVE
vendor rg (the honest wall-delta=instruction-delta bar): silesia-isal T8 = gz 379 vs rg 324 =
~1.17x REAL LOSS — banked F1 "win-at-parity" RE-OPENED; ALL near-bar cells (bignasa 0.986,
weights 1.03, T1 wins mostly survive at ~5% tax share) need RE-SCORING vs native rg. ACTION
OWED: parity.sh points at the native vendor build (or dual-reports, wheel labeled startup-taxed);
matrix re-score session. LEDGER PASS disposition: corpus was SILESIA not model (strike 4,
corpus-vs-brief class) — model narrative VOID; BANKED silesia-labeled: ledger method; finalize =
23% of finish_oracle, O(output)-shaped (zero on no-prefix C8), +15.5ms T8-ideal vs rg alloc+copy
=> O(N) prepend_bytes HYPOTHESIS w/ pre-registered Finalize perturbation (3ms sleep/chunk, CRC
control) — must run on BOTH silesia and model vs the NATIVE bar; ffi_outside bimodal = cache
hypothesis; commit=0; gz ISA-L kernel FASTER than rg's (65.4 vs 71.5 ideal, silesia-specific).
STRIKE-5 CLAUSE adopted for all briefs: step-0 prints BRIEF_CELL=<corpus:T> + input sha; mismatch
with the brief's stated sha = abort CELL_MISMATCH; reports lacking the pair are void on arrival.
## BOOTSTRAP CRITICAL ON MODEL-ISAL T8 (first marker-loop criticality CONFIRMED) — inc-1 = N2-N5 Rust ladder [2026-06-12]
probe/bootstrap-isal (054db213) gated. POSITIVE CONTROL: the isal build runs the bootstrap
(header 3.8ms/2% + BODY 155ms/98% thread-summed, 7.68MB at 50MB/s — 155 not 270: keepIndex
deleted per-backref sparsity tracking INSIDE the body; footnote not load-bearing). PERTURBATION:
sleep-surviving +254ms at 15.9x inject, proportional 2.10 (spin 5.7x-inflated = core-blocking
structural artifact, correctly discarded) => the FIRST cell where marker-loop criticality is
CONFIRMED (coexists with the bignasa/silesia refutations — different slack structure; no
contradiction). CEILING: ~16-19ms SLOPE-EXTRAPOLATED of a 63ms-this-session/49ms-banked gap
(~15ms denominator drift band). WORKER ERROR CORRECTED AT GATE: "native T4 ABSENT (generalizes
from T1)" is WRONG — T1's mechanism is the zero-thread inline pool (T1-only); native T8 routes
~75M/211.9M bytes window-absent (asm-campaign §9) => native T4+ = OPEN. Matrix corrections
appended to engine-w-refresh.md (0b054ef5). INC-1 RULING: (i) N2-N5 Rust ladder on the marker
fast loop [DISPATCHED — F-w1: ship >=2%, TIE=>keep-if-byte-exact + bank body-rate counter; rate
>=1.5x w/ flat wall => slope-ceiling falsified, bootstrap track CLOSES]; (ii) bootstrap asm
DEFERRED (rung-d entry conditions; realistic increment single-digit ms); (iii) PARALLEL ledger
pass on the UNNAMED ~47ms residual [DISPATCHED — first action: commit the uncommitted
/tmp/gz-combined instrument tree; refresh decodeBlock ledger post-keepIndex; attribution-only
pending perturbation]. Box serialized; builds/gauntlets are not.
## ENGINE-W INC-0: bootstrap STRUCTURALLY ABSENT at native T1 — inc-1 dead there; 270ms re-attributed to ISAL build [2026-06-12]
probe/engine-inc0 (2 commits; GZIPPY_SLOW_BOOTSTRAP knob built, per-byte in marker_decode_step_loop,
DUAL-SHA). FLAT BY CONSTRUCTION at native T1: the bootstrap path receives ZERO bytes on all 5
corpora — advisor STRENGTHENED the mechanism (worker's stored-preamble story WRONG): chunk 0 gets
a stack zero-window (chunk_fetcher.rs:2620-2625) + bit-0 sentinel (:582-585), AND at T1
pool_threads=0 (:663, vendor BlockFetcher.hpp:185) => tasks run inline-deferred AFTER the
predecessor published => window_map.get always hits. WINDOW-ABSENT IS IMPOSSIBLE AT NATIVE T1 BY
CONSTRUCTION (corpus-independent). T1-ONLY: native T4+ bootstrap stays OPEN (real threads race
the publish). Bootstrap asm (inc-1) DEAD at native T1 per pre-registration; may be REBORN on
model-isal ONLY if the redirect probe shows criticality. RE-ATTRIBUTION BANKED: the rdtsc "270ms
bootstrap/prefix" = the ISAL build's speculative window-absent phase (decode_chunk_window_absent
:3636 -> unified marker loop — the knob sits in EXACTLY that loop on the isal build; verified no
asm-kernel-disable confound). Fresh T1: silesia native 929 vs rg 824 = 0.888x; model 0.772x —
worker's "gap entirely in seeded clean path" = ATTRIBUTION (provisional; oracle owed). REDIRECT
RATIFIED (b)->(a)->(c): (b) calibrated bootstrap perturbation on model-isal T8 [DISPATCHED:
positive control BOOTSTRAP_BODY_BYTES>0 else STOP; calibrate spin/byte; arms OFF/50/100/sleep;
ceiling pre-registered ~34ms wall vs ~50ms cell gap; FLAT => bootstrap track CLOSED everywhere];
(a) T1 seeded-decode removal oracle (prices the kernel grind; Oracle-C degeneracy warning);
(c) kernel grind gated on (a) + plateau evidence. engine-w-refresh.md matrix corrections owed
with (b)'s result.
## KEEPINDEX PORT MERGED (c31d9a07) — model-isal 1.19x -> ~1.13x; sparsity CPU deleted; RSS-neutral [2026-06-12]
fix/keepindex-config (f60e8e3b) merged on MERGE-NOW. Port is VERBATIM vendor (only the two fields
flip at keepIndex=false; sm_driver is the sole production constructor; Default keeping
sparsity=true is itself faithful — vendor flips at the reader-config layer). Gate verified in
source: nothing downstream needs masked windows (full window = strict superset; successor seeding
via last_32kib_window unaffected); old path zlib-compressed EVERY stored window — raw-uncompressed
at keepIndex=false is exactly rg's CLI-default behavior. RSS measured ON-BOX: TIE (model 294.0 off
vs 292.8 on MB; silesia 274.6 vs 270.0; deltas < spread; worst-case exposure 1.6MiB) — the
pre-existing ~2.7x model RSS gap vs rg (287 vs ~104MB) is UNTOUCHED and remains the separately-
owned weak column. RESULTS: 80/80 sha grid; TIE-or-WIN all cells (model T4 1.039 off-faster,
silesia T8 1.032 off-faster); 430ms thread-summed CPU deleted (=~10ms T8 wall — parallel-region
CPU; compounds at low T). NEW model-isal standing: T8 ~424ms vs rg ~372-375 = ~1.13x (was 1.19);
REMAINING NAMED DELTA = bootstrap/marker-prefix ~270ms = ENGINE-W TRACK (the funded rewrite is
now the next and only named lever for that cell). Non-blocking note: decode_bypass replay uses
ChunkConfiguration::default() (sparsity=true) — replay config now differs from production; one-
line comment owed. Slope-recal skipped (brief cited a Dict knob that never existed in Rust).
LOSS SURFACE NOW: silesia T4/T16 (NEEDS NEUROTIC REBOOT), model-isal residual (engine-W),
native T1/low-ratio (engine-W), RSS column (separately owned). Plus the release-correction PR
awaiting user nod.
## LEDGER CLOSED ON-BOX — model-isal gap NAMED: window-sparsity decode rg SKIPS at keepIndex=false (sm_driver.rs:127 divergence) [2026-06-12]
Dispatch-silence probe gated; advisor ran a COMBINED instrument on the guest (both probes' counters
one binary, frozen, sha-exact, TSC-calibrated) and reconciled the cross-probe contradiction exactly:
decodeBlock CPU 3,018ms = kernel 2,198 + wrapper 42 [refill-staging moved the 128KiB memcpy here —
explains 3.6->42 across probes] + CRC 35 + boundary 33 + FINALIZE-SPARSITY 430 + bootstrap/prefix
270 + ~10. BANKED: (a) the stale "53ms dispatch silence / 5 stalls" note RETIRED (fixed by the
hit-drive port; kill-switch A/B +6.5ms FLAT at HEAD); (b) the probe worker's "consumer is
pace-setter" REJECTED — ttp.rx_recv_block (74% of consumer wall) is the wait-for-worker-delivery
span; WORKERS pace; FLAT slow-knobs only prove the knob-covered pure-Rust loops slack (the FFI
bulk + finalize were never injected); (c) the "buffer lifecycle / DecodedData single-pass fix"
localization REFUTED by measurement (truncate/commit = 0.0ms; those components ~68ms) — fix
MIS-AIMED, not dispatched. THE NAMED DIVERGENCE (X1 = 430ms = 2/3 of the gap):
finalize_window_for_last_subchunk -> get_used_window_symbols = a 32KiB pure-Rust marker-engine
decode + backref tracking PER CHUNK (~8.5ms x 51) — vendor runs the identical routine ONLY when
keeping an index (GzipChunk.hpp:61-97; ParallelGzipReader.hpp:1330 windowSparsity = m_keepIndex &&
m_windowSparsity; CLI default keepIndex{false}, rapidgzip.cpp:47,167); gzippy HARDCODES
window_sparsity: true (sm_driver.rs:127). X2 = 270ms bootstrap/marker-prefix = engine-W track.
DISPATCHED: faithful keepIndex port (delete-divergence: plain-decode CLI => keep_index=false =>
window_sparsity=false + window_compression_type=Some(None), mirroring applyChunkDataConfiguration
ParallelGzipReader.hpp:1326-1334; sparsity path kept for future index mode) + falsifier (model
T4/8/16 vs rg, expect tens-of-ms or TIE-KEEP) + dict-site slope recalibration at HEAD (the banked
6.99ms/ms predates the hit-drive fix). Box note: stray gzippy-07c1cb9d test binary (fd_vectored
class) idling 2.4% CPU — kill before banking absolutes. Combined-instrument tree preserved at
/tmp/gz-combined (uncommitted).
## STAGING+GUARDS MERGED (a53c53e0) — refill staging TIE-KEEP; flavor guards live; fd_vectored hang = PRE-EXISTING env [2026-06-12]
fix/isal-refill-staging merged on gate PASS. (1) refillBuffer 128KiB staging port (isal.hpp:163-205,
chunk clean-tail site only, T1 single-shot untouched): DUAL-SHA exact, falsifier TIE (435 vs 434ms
best-of-7) => TIE-KEEP per rule 7 (the last named FFI-surface divergence is now closed). (2)
STORED_DEMOTE counter dumped in VERBOSE + unit fixture pins the <50% demote gate. (3) FOOTGUN
GUARDS LIVE: BUILD_FLAVOR in --version + first GZIPPY_DEBUG line; build.rs warning on
legacy-serial; _parity_guest.sh HARD-FAILS before timing on flavor mismatch (demonstrated both
directions). (4) AUDIT BANKED: release.yml ships LEGACY-SERIAL binaries for ALL targets today
(isal-compression does not set parallel_sm — build.rs:94); ci.yml/benchmarks.yml same; Cargo
default-flip BLOCKED by 8 pre-existing raw-pointer clippy errors + 45 warnings in the parallel
module => release-correction PR = explicit per-target features NOW (x86_64: gzippy-isal; arm64:
pure-rust-inflate — gate measured arm64 parallel 2x FASTER than legacy at default-T on
compressible, banked 16x pathology does NOT manifest, StoredParallel catches it; forced-T1 1.85x
slower = pre-existing classifier policy), lint-debt cleanup + default-flip deferred. SUITE-HANG
DISCRIMINATED: all 8 fd_vectored_write (pipe/vmsplice) tests hang IDENTICALLY on base AND branch
on the guest => PRE-EXISTING environment change (these passed green days ago — box state changed,
possibly post-outage; ENV WATCH-ITEM, not a code defect; also killed a 20-HOUR deadlocked
gzippy-bar2 test binary squatting since a stale session). NEXT: Worker B dispatch-silence probe
(model-isal residual); release-correction PR (user-visible — confirm before tagging).
## ISA-L OVERHEAD HYPOTHESIS REFUTED (direct rdtsc) — model-isal residual UNNAMED; staging port + guards dispatched [2026-06-12]
Probe (probe/isal-overhead 0460fcdf, sha/routing-asserted, FFI-surface instrumented): the "isal
uses ~40% more CPU than rg's ISA-L on model" claim was an ESTIMATION ARTIFACT — direct rdtsc:
isal_inflate kernel 2,128ms thread-summed vs rg's banked 2,064ms = small single-digit-% (CAVEAT:
rg side is a banked figure from a different instrument — the 40% refutation is robust, the exact
3% is not); wrapper 0.1%, CRC 1.4%, boundary 1.5%. FFI surface structurally vendor-identical
(1 init + 1 set_dict 32KB + ~131 calls/chunk) EXCEPT ONE divergence: input staging — gzippy
passes the whole remaining chunk as avail_in (avg 2,095KB, L3) vs rg's refillBuffer 128KB L2
staging (isal.hpp:163-205). Causal: per-chunk decode IS on the T8 critical path (dict-site sleep
slope 6.99 vs predicted 6.375 ms/ms, +/-10%). Worker's closing "gap = banked ~40% HoL" REJECTED —
that finding is RETIRED; model-isal residual (~65ms/444 = 1.19x worst-best, NOT drift-band) is
UNNAMED; the one model-specific lead = the 53ms dispatch-silence (chunk_fetcher.rs:1515 area) +
pool-eff 86.3 vs 89.2. DISPATCHED (advisor-ordered): Worker A = faithful refillBuffer staging
port SCOPED to the chunk clean-tail FFI site ONLY (T1 single-shot untouched — banked WIN surface,
staging sign unknown there; separate commit if ever) + storedheavy nits (counter dump, demote
unit fixture) + FOOTGUN GUARDS (BUILD_FLAVOR string in --version/GZIPPY_DEBUG from emitted cfgs;
parity/measure-script hard assert on flavor; build.rs warning when neither product feature set) —
fail-closed at the script layer is the actual strike-4 prevention. Cargo default-flip to
pure-rust-inflate = separate PR after CI-lint-gate check. Worker B (after A): dispatch-silence
probe, perturbation pre-registered. ANOMALY BANKED: --features isal-compression alone = silent
legacy serial binary (no parallel_sm cfg) — cost 90min; build.rs comment says default stays []
only due to parallel-module lint debt.
## STOREDHEAVY CLOSED + MERGED (c78f98b1) — routing demotion flips 1.53-1.60x LOSS to 0.50-0.65x WIN [2026-06-12]
fix/stored-routing 36d04ca0 merged on advisor MERGE-NOW. The demotion (stored_split.rs:326-355,
WalkEnd::HuffmanTail arm: prefix_out>0 && prefix_out*2 < expected_size => NotStoredDominated =>
dispatcher falls through to ParallelSM) is CLEAN at the edges (no pre-demote output — the walk is
a pure scan; original data slice re-parsed from byte 0; overflow-safe; ISIZE-lies => old behavior;
boundary strict-<; multi-member unreachable). VENDOR: rg has NO stored route at all — demotion
moves gzippy TOWARD vendor. FALSIFIER PASS with 2x margin: storedheavy T4/T8/T16 0.50-0.65x vs rg
(was 1.53-1.60x); kill-switch + counter verified; no regression (storedmix/monorepo ~1.000 A/B,
pure-stored takes WalkEnd::Final, isal T1 untouched). ADVISOR CLOSED THE ISAL GAP ITSELF on the
guest: gzippy-isal build, storedheavy T4/T8 demote-on/off ALL == pin d3efb3ac, demote+routing
prints verified, storedmix/monorepo/pure-stored byte-exact vs zcat. PERF: unfrozen absolutes
UNBANKABLE (rg 77-81 vs banked 94-100 = box-state both directions) but the 2.9x structural A/B
direction stands; post-merge frozen refresh re-certifies. "Beats rg 2x" physical: incompressible
tail = literal-decode race; T8 2.57GB/s aggregate / T1 1.48GB/s credible for the BMI2 multi-lit
fastloop; rg scales only 1.6x here (its finder pays on dense dynamic-Huffman-over-random).
NON-BLOCKING NITS for a follow-up: STORED_DEMOTE counter is increment-only (doc says dumped);
no unit test pins the demote gate (e2e-covered; add a <50%-stored fixture test).
LOSS SURFACE NOW: model-isal per-chunk ISA-L overhead (1.19x), silesia T4/T16 (drift-gated,
NEEDS NEUROTIC REBOOT), native T1/low-ratio (engine-W funded track). Everything else WINS.
## POST-P0 REFRESH — monorepo WINS all cells; storedheavy = ROUTING MISFIRE (named); model/weights = clean-tail economy [2026-06-12]
(measure/post-p0-refresh c68ac933; artifact shas printed per the strike rule; routing asserts per
cell; STALE_CONFIRMED_BLOCK_SKIP = 0 everywhere, sha-OK everywhere — the P0 fix HOLDS.)
MONOREPO: formerly RC:1 — now WINS every cell both builds (native 0.83-0.95x; isal 0.62-0.96x).
STOREDMIX: wins all cells (0.59-0.86x). STOREDHEAVY MECHANISM FOUND (the named lead pays off):
first_block_is_stored sees the 8.2MB stored PREAMBLE (262 blocks, 8.2% of output) =>
StoredParallel; but the file is 91.8% dynamic-Huffman tail which StoredParallel decodes
SEQUENTIALLY (lut_bulk_inflate ~94ms flat at every T — phase timing: overlap_copy+tail = ~100% of
wall) while rg speculatively parallelizes it (24 chunks, 0 false positives, 95ms wall at T8 = the
tie target). isal T1 wins big (0.598x IsalSingleShot) then T4+ collapses to 1.53-1.60x on the
route switch. WOULD-BE FIX (dispatched): post-walk check — stored prefix < 50% of output =>
NotStoredDominated => ParallelSM. MODEL/WEIGHTS ECONOMY (T8 captures + traces both tools):
speculation is NOT the problem (96.2% accept, 51/51 flips, pool eff 86-89% ~ rg's 89); the cost
is CLEAN-TAIL decode: native pure-Rust clean ~42% more CPU than rg's ISA-L (= engine-W, the
funded track — model/weights native cells are ENGINE cells, not scheduling); isal's ISA-L usage
~40% more CPU than rg's ISA-L on model [ESTIMATION ARTIFACT — REFUTED 2026-06-12 by direct rdtsc:
kernel ~parity, see the ISA-L-overhead probe bank entry above] (isal model T8 1.192x is the
remaining isal loss); weights isal T8/T16 = 1.030/1.025 NEAR-BAR. Pre-registered clean-tail
perturbation design banked in the worker report (GZIPPY_SLOW_CLEAN_PCTL + skip-oracle ceiling).
FULL matrix rows updated in the branch file. NEXT: (a) storedheavy routing fix [dispatched,
falsifier: T4+ flips to ~rg-class, no regression storedmix/monorepo/pure-stored]; (b) isal model
per-chunk ISA-L overhead probe; (c) engine-W owns model/weights-native.
## P0 CLOSED + MERGED (b879cc4d, advisor MERGE-NOW) — ring word-copy overshoot clobbered the oldest undrained slot [2026-06-12]
Round-2 hunt (Fable) root-caused the 1-byte corruption: emit_backref_ring's non-overlap word-copy
arm rounds length to 4 u16 slots ((len+3)&!3; u8 twin rounds to 8 at ~:4153) — the rounded
overshoot can WRAP the ring onto the FIRST UNDRAINED slot of a maximally-full read() call (cap
65278-1 + a 258-len backref at the exact call boundary), clobbering one already-decoded byte
before the end-of-call drain. Vendor NEVER overshoots (exact memcpy, deflate.hpp:1376) — this was
a gzippy deviation whose "tail overwritten before read" assumption is false at call boundaries.
FIX (41c9c9be): gate the word arm on (*pos - drained) + rounded <= RING_SIZE (both twins);
guard-fail falls to exact-length arms. Gate verified the guard sound at edges (drained<=pos on
every path incl. resumable re-entry + flips; ZERO conservatism — fires only when undrained span
>= 65276/65536) and waived a wall A/B (ring emitters only, marker path causally slack; contig +
asm kernel untouched; NOTE for the instruction ledger: +1 sub/add/cmp per ring backref).
EVIDENCE: 3 fail-at-parent regression tests (incl. hand-built stream reproducing the geometry at
index 65278); mono-gnu9 sha-exact T1-T16; guest grid 56/56 both builds; STALE_CONFIRMED_BLOCK_SKIP
= 0 on this stream (98fd618c's path uninvolved — TWO distinct bugs, both now fixed). The
"stored-flip 51542" trace line was a RED HERRING (stream has zero stored blocks; fired in a
discarded trial decode at a false candidate offset). MERGED: 98fd618c + 9dce0b8f + 41c9c9be +
4f252fad (record-correction docs: GzipBlockFinder has NO scanner — the raw block finder
(block_finder.rs) is the component that hands stored-payload false positives to trial decodes).
WATCH-ITEM filed (advisor falsifier): stored-heavy corpora w/ --verbose — STALE_CONFIRMED_BLOCK_
SKIP>0 with sha-exact = guard benign; any CRC mismatch or published-chunk offset matching a
never-confirmed raw-finder candidate = trial-decode leak is real. NEXT: refresh the FAIL(RC:1)
matrix cells (monorepo now passes), then move 3 (model/weights low-ratio economy).
## HUNT WORKER'S "P0 CLOSED" VOIDED — supervisor re-ran the real reproducer: STILL FAILS; artifact-confusion strike 3 [2026-06-12]
The hunt worker declared the P0 closed by 98fd618c — but its sanity line tested
/tmp/monorepo.tar.gz (9,822,456 B = the APPLE-gzip stream, which the gate already proved 98fd618c
fixes) NOT the preserved reproducer /tmp/mono-gnu9.tar.gz (9,819,846 B, sha 5f6cc8ee — GNU gzip
1.14 -9). SUPERVISOR RECONCILED FIRST-HAND: fix build at -p 4 on mono-gnu9 => RC=1 CRC mismatch,
the IDENTICAL one byte (offset 35335338, 0x4C vs 0x2E). The gate's result stands; the worker's
verdict is VOID. THIRD artifact-confusion incident (native-as-isal, default-as-isal, apple-as-gnu)
=> NEW BRIEF RULE: step 0 of any reproducer work = print + assert the input sha256; analysis on
any other artifact is void by construction. STATE: 98fd618c fixes the APPLE-stream/banked-class
failure (false-positive confirmed block decoded as garbage — its data_with_markers[890657]=0x4C
probe evidence is real FOR THAT STREAM) and stays VERIFY-THEN-MERGE; the GNU-stream 1-byte bug is
a DISTINCT mechanism (all accepts exact-offset; skip-guard irrelevant) and the P0 REMAINS OPEN.
Worker's Part-B out-of-sync counter (STALE_CONFIRMED_BLOCK_SKIP, vendor-faithful position) is
UNCOMMITTED in /tmp/gz-crc — salvage it in the next pass.
## P0 FIX GATED: VERIFY-THEN-MERGE as hardening — but the P0 SURVIVES (distinct 1-byte bug, reproducer banked) [2026-06-12]
Fable advisor BUILT both commits and empirically refuted the fix-closes-P0 claim: a GNU gzip 1.14
-9 stream of the same monorepo.tar (/tmp/mono-gnu9.tar.gz, sha 5f6cc8ee..78e8c71, preserved local
with /tmp/monorepo.tar + trace /tmp/sm-gnu9.log) STILL CRC-fails T>1 both builds IDENTICALLY pre/
post-fix: deterministic ONE-BYTE corruption (offset 35335338, got 'L' want '.'), same stored-flip
case1-ge-window 51542 signature, NO_STORED_FLIP no effect, all 9 speculative accepts exact-offset
(not the acceptance-range path). THE LIVE P0 = a pre-existing 1-byte marker/window resolution bug
on stored-bearing data. The 98fd618c skip-guard change itself is SAFE hardening (verified: only
decoded-subchunk ends are confirmed-inserted (:3122/:3125) => confirmed entries never lead the
frontier; skip drops only redundant work, emission contiguous from 0; BGZF/StoredParallel don't
route here) but its commit narrative is FALSE (no "GzipBlockFinder raw scan" exists — the phantom-
confirm mechanism is undemonstrated) and its test PASSES ON THE PARENT (240KiB fixture = single
chunk < MIN_ADJUSTED_CHUNK_BYTES 512KiB — the skip path never fires). VENDOR: rg cannot need this
skip — guesses strictly past last confirmed (GzipBlockFinder.hpp:286) and it ASSERTS out-of-sync
("Next block offset index is out of sync!", appendSubchunksToIndexes) — the skip is a DIVERGENCE
(silent-resync vs fail-loud) to document, with rg's out-of-sync check as the faithful complement.
ALSO FLAGGED (own falsifier owed later): acceptance-range divergence — gzippy claims
[partition_seed, decode_start] (:3686-3688) vs vendor candidate-pair-only (GzipChunk.hpp:721-722);
exonerated by trace on THIS artifact. MERGE CHECKLIST (dispatched): amend message/test-doc; add
rg's out-of-sync debug counter (:1977); true pre-fix-failing fixture or relabel smoke; guest grid
with the BANKED artifact both builds; THEN merge + hunt the live 1-byte bug with the local
reproducer. P0 REMAINS OPEN.
## ISAL COLUMN RERUN (correct build) — F1/F2/F3 ALL PASS; gz-isal wins T1 on EVERY corpus; 32W/9L vs rg [2026-06-12]
Rebuild verified BEFORE measuring (path=ParallelSM @T8, path=IsalSingleShot @T1 — the asserts the
voided session skipped). All three pre-registered falsifiers PASSED: F1 silesia T8 gz 340 vs rg
350 = win-at-parity (voided session's 810 flat row proven misbuild); F2 gz-isal T1 == igzip +/-
10ms on all 11 corpora, margin NOT size-scaling (corr -0.40) — the single-shot wrapper costs ~0;
F3 scaling curves restored (model 2020->360ms T1->T16). THE CORRECTED PICTURE (banked,
measure/field-matrix 1f14e4b7): gz-isal BEATS rapidgzip at T1 on ALL 11 corpora (0.36-0.91x) and
wins 32/41 measurable cells overall. THE REAL LOSS SURFACE (named rows): storedheavy T4-T16
(1.25-1.45x — parallel SM overhead on pure-stored; NOTE a StoredParallel route EXISTS at
mod.rs:218-224, why didn't it fire/win? named lead), model T4-T16 (1.16-1.26x — the low-ratio
economy gap, move 3), silesia T4/T16 (1.10-1.11x — the drift-limited terminal cell + T16),
weights T4 (1.12x). Defaults audit: bare gzippy wins/ties 7/10 vs bare rapidgzip (model +
storedheavy lose). monorepo FAILs RC:1 at T4+ on the isal build too — P0 is cross-build
(stored/marker path is shared); T1 passes. WORKER-PHRASING CORRECTED AT SOURCE: small corpora
flat at T>1 is CHUNK GEOMETRY (<=4MiB compressed = 1 chunk), NOT a "routing threshold"
(mod.rs:208-224 has no size gate). CAMPAIGN REFRAME this enables: the goal surface is now 5
named loss families — (1) P0 stored/marker CRC [fix in flight], (2) storedheavy stored-path
T>1, (3) model/weights low-ratio economy, (4) silesia T4/T16 band, (5) native T1 engine rate —
everything else WINS in product configuration.
## FIELD MATRIX GATED — isal column VOID (misbuilt binary); P0 CORRECTNESS BUG found (monorepo ParallelSM CRC) [2026-06-12]
First field measurement (measure/field-matrix, plans/field-matrix-2026-06-12.md: 11 corpora x T x
{gz-isal, gz-native, rg 0.16, rg-upstream, pigz, igzip, libdeflate, gzip}, N=5 sha-verified).
Fable gate verified the suspicious rows LIVE on the guest:
ARM VOID: the gz-isal binary was built with DEFAULT features (Cargo.toml default = [] => no
parallel_sm, no ISA-L) = the LEGACY SERIAL LibdeflateSingle build (verified path=LibdeflateSingle
threads=8; RSS == libdeflate +/-1MiB on all 11 corpora). ALL gz-isal cells void. The worker's
"zero scaling = 10MiB routing threshold" claim REFUTED: NO threshold exists at HEAD
(MIN_PARALLEL_COMPRESSED dead code) — it re-derived the phantom from CLAUDE.md's STALE routing
table (now corrected in CLAUDE.md). Mislabeled-binary scar repeat #2 — supervisor brief said
"isal default", which was wrong; binary verification (GZIPPY_DEBUG path assert) is mandatory in
every future brief.
P0 CORRECTNESS BUG (real, reproduced): gzippy-native monorepo.tar.gz (9.8MB single-member,
stored-block-bearing) exits RC:1 at T>1: CRC32 mismatch (expected 8c7ca615 got 0e0efdd5) after
"stored-flip case1-ge-window uncompressed_size=51542"; T1 clean. GZIPPY_NO_STORED_FLIP=1 does NOT
fix => bug is in the wider stored/marker speculative interaction, NOT confined to
try_read_stored_special (marker_inflate.rs:1369-1419). Repro: GZIPPY_DEBUG=1 gzippy-native -d -c
-p 4 /tmp/squishy/monorepo.tar.gz. Rule 4: this outranks all perf work.
VALID native rows BANKED: wins/ties T8-T16 big-compressible (silesia 0.97-1.00, bignasa
0.92-1.06, nasa 0.88-0.93 — BEATS rg outright at T16 on bignasa/nasa); loses T1 everywhere
(1.38-2.0x, known engine rate); loses model/weights ALL T (1.14-1.70x) with CPU-s/GB +55-66% vs
rg — LOW-RATIO CORPORA are the biggest valid non-T1 gap (too big for kernel rate alone; suspect
excess speculation-failure/stored work); storedheavy 1.36x flat; storedmix WINS 0.64-0.67.
P1 falsified on valid rows alone, but 58/27 contaminated — recount after isal rebuild. P2
falsified ONLY vs rg (~19 native cells; vs pigz = the speculation-architecture class rg fails
too — pigz decodes serially at 4.6 CPU-s/GB). P3 (RSS) confirmed for native. igzip = T1
field-best kernel; product-isal-vs-igzip margin unmeasured (wrong binary).
RANKED NEXT: (1) P0 stored/marker CRC fix; (2) isal rebuild --features gzippy-isal + re-run
column/defaults-audit (falsifiers: path asserts, silesia T8 ~ rg parity, T1 within constant ms of
igzip else open structural diagnosis); (3) model/weights low-ratio economy gap (fulcrum vs ->
perturbation). DEFERRED: consumer-busy delta (below #3, may share mechanism); T16 fallback curve
(GAP-2) DEPRIORITIZED — T16 is gzippy's best valid column.
## WAIT-PUMP DEAD + saturation hypothesis REFUTED-AT-SOURCE — silesia T4 banked DRIFT-LIMITED TERMINAL [2026-06-12]
The first legal divergence (lever/wait-pump f951bd6b: 1ms prefetch pump inside the consumer's
recv_post_process_blocking wait; kill-switched, effect-counted, 36/36 sha grid, no regression on
8 net cells) is DEAD per its pre-registered falsifier: silesia T4 median on 591 vs off 581ms
(bar >=25ms). Counters told the story: drive_invocations=66, tasks_submitted=0 — every pump tick
found the pool LEGITIMATELY full. Fable advisor REFUTED-AT-SOURCE my saturation-accounting
hypothesis: gzippy harvests ready futures BEFORE gating (block_fetcher.rs:729-739 ==
BlockFetcher.hpp:463-470), cap P-1 in-flight + 2P cache both vendor-exact (chunk_fetcher.rs:578-579
== BlockFetcher.hpp:181-185); the 3 counted decodes were genuinely running (3 decodes +
1 post-process task = 4/4 threads). rg ALSO has no prefetch driver during consumer-busy (its
post-process wait is a bare future.get, GzipChunkFetcher.hpp:517) — the QUEUE-EMPTY gaps are a
property of the SHARED architecture; the slow-consumer +33.9% slope = lengthening driver-less
serial sections, shared-architecture property, not a gzippy defect. BANKED: silesia T4 isal
queue-empty starvation is STRUCTURALLY FAITHFUL; the residual 3pp efficiency delta = rg's SHORTER
consumer-busy phase per chunk (post-process queueing/harvest/write bookkeeping instructions) — an
instruction-delta-to-locate-and-converge item (fulcrum vs per-chunk consumer-busy comparison),
queued behind the field matrix. STATUS: silesia T4 = DRIFT-LIMITED TERMINAL pending field matrix
+ drift repair (neurotic reboot still owed by user). Increment-1 DISPOSITION: DISCARD unmerged —
vendor has no pump at that wait, the change is a no-op by construction; branch lever/wait-pump
stays as the falsification artifact (a "deeper cap" second divergence is mechanically available
but FORBIDDEN-until: fulcrum vs locates the consumer-busy delta first; vendor explicitly bounds
pool tasks <= P at BlockFetcher.hpp:564-568). NEXT: reflection Move 2 — the FIELD MATRIX +
efficiency bank session.
## DISPATCH-STARVATION CONFIRMED (super-proportional falsifier) + FABLE REFLECTION ADOPTED — divergence LEGALIZED [2026-06-12]
TWO landmark results. (A) MECHANISM PROBE (probe/parallel-eff 80a6670d/f3b0c050, frozen, sha-OK):
ALL idle gaps are QUEUE-EMPTY (10/8/7 gaps across 3 runs, 128-182ms total, others_active=3 in
every gap) — workers starve because the consumer, blocked in recv_post_process_blocking (~5ms/chunk
apply-window wait, 100us poll loop), never drives prefetch_new_blocks; the pool is a pure Condvar
queue (workers need no consumer once a task is queued). PRE-REGISTERED FALSIFIER CONFIRMED:
GZIPPY_SLOW_CONSUMER sleep +50% -> +8.5%, +100% -> +33.9% — SUPER-proportional, 7x/14x the 13ms
null slope. STRUCTURAL DIFF VS VENDOR: NONE — rg ALSO blocks at future.get (GzipChunkFetcher:516)
without prefetching; the 3pp efficiency delta is QUANTITATIVE (per-chunk post-process duration),
not structural. Under the old unlike-rg prohibition this was a dead end; under (B) it is a LEVER.
Also: fulcrum locate now runs unflagged-wall but FLAGS a 19.1% dark zone = finalize_with_deflate
(10.5%) + boundary/CRC lack trace_v2 spans in run_post_process_task — scan_candidate #1 (56.4%
on-path) still CANNOT be confirmed or killed until the dark zone is spanned; chunk_id=MAX Gantt
bug FIXED. (B) FABLE REFLECTION (user-directed full reflection; advisors are FABLE from now on).
ADOPTED: (1) DIVERGENCE FROM RG LEGALIZED — allowed when byte-exact + causally verified at the
wall on >=1 cell w/ no regression cell + recorded in a DIVERGENCE LEDGER w/ vendor counterpart;
rg = blueprint + oracle, NOT ceiling (the campaign's biggest wins were already divergences).
(2) Tiered gates: bank/merge/direction-change get advisor; falsifier-armed probes self-gate.
(3) Bar: pooled >=3-session 95%-CI >= 0.99 REPLACES worst-session, gated on drift repair (ASK
USER: reboot neurotic — 53d uptime; post-reboot band <=1.5% or 2nd box). (4) Field matrix owed:
{rg 0.16 + upstream, pigz, igzip, libdeflate, gzip} x Squishy x T + defaults-audit cell; bar =
max(all tools); bank CPU-s/GB (already <= rg = a claimable efficiency title) + peak RSS (+21-25%
vs rg = open item). (5) Re-synthesis obligation (the stale 0.667x anchor leaked into briefs after
rung-c shipped — MEMORY patched this turn). (6) Compression = abandoned-by-drift (zero commits
since 2026-05-25): owe a baseline matrix then a dated user decision. (7) Owned leads: T16
fallback-rate curve (GAP-2: fallbacks>=chunks at T16!), model 0.857 own decomposition.
NEXT: implement the FIRST LEGAL DIVERGENCE — pump prefetch from the consumer's blocked-wait poll
loop (and measure); falsifier: gaps <=10ms/worker AND >=half-gap ratio recovery, else lever dead
and silesia T4 = drift-limited terminal.
## RESUMED PROBE GATED — CPU gate CONFIRMED; idle DISPATCH GAPS named; ISA-L "recoverable" claim discarded (Rule 3) [2026-06-12]
Box back (network outage, host uptime 53d — never rebooted). Frozen session, sha-OK, N=10:
wall gz 563 / rg 498 = 0.911x. BANKED: (1) CPU GATE CONFIRMED 2nd session — gzippy thread-summed
decode CPU 1825ms <= rg 1941.6ms (6% less/byte); finalize_with_deflate's 184ms is INSIDE that
passing total. (2) rg ThOpt=358 RECONCILED (advisor, vendor source): subset numerator
(decodeBlockTotalTime+applyWindow+checksum)/P — NOT the full verbose table/P (485); no two-pass
double-count (applyWindow resolves in place, ChunkData.hpp:302). Efficiency banks ONLY
formula-matched: gz 93.7% ~= rg 93.6%, with the HARD CAVEAT that raw 88.4% is the gap-honest
figure — equal pool-packing does NOT imply wall parity; the +5.3pt "correction" pads exactly the
idle-gap region. (3) GANTT four answers: 172ms marker-only tail chunk EXONERATED (finishes
~454/502); pool-fill INSTANT (<1.3ms); NO start-time window wait (all 16/17 chunks decode
window-absent speculatively; window needed only for apply_window); MID-RUN IDLE DISPATCH GAPS
~40ms/worker (160ms thread-summed, largest 29.6ms) = wall-above-ideal 50-63ms = THE NAMED
CANDIDATE. (4) finalize_with_deflate 184ms = CPU-attribution discovery; it is a FAITHFUL PORT of
rg finalize/cleanUnmarkedData (ChunkData.hpp:418-421, GzipChunk.hpp:155-158) whose rg cost is
BURIED inside decodeDurationIsal — the worker's "184 vs rg-append-48 = +136 divergence" is
INVALID (different ops); watch-item only. DISCARDED: "+70ms recoverable / ISA-L 2x => beats rg"
(Rule-3 slope->ceiling on unspeedable vendor asm); the "paradox" paragraph (apples-to-oranges
ideals). UNCONFIRMED (cheap confirm-or-kill queued): locate's scan_candidate #1 (274ms on-path,
from a correctly-SELF-FLAGGED run — wall-window mismatch; locate's first production self-flag
worked). OPEN MECHANISM (the faithful-port question, next probe): is gzippy DISPATCH-DRIVEN
(worker idles until consumer processes the finished chunk) where rg is PREFETCH-DRIVEN
(BlockFetcher keeps the pool fed, BlockFetcher.hpp:111-114)? Falsifier: GZIPPY_SLOW_DISPATCH
inject into consumer post-chunk processing, sleep-primary. Anomalies: Gantt chunk_id=MAX for
window-absent (instrument bug); residual sub-conservation 85-87% (37ms unaccounted); isal_chunks
14/17 (3 never flip); guest lacks bc.
## PARALLEL-EFF PROBE BLOCKED — NEUROTIC UNREACHABLE; rg --verbose semantics RESOLVED from vendor source [2026-06-12]
The parallel-efficiency probe (probe/parallel-eff, 27b1ffcf) completed ALL instrumentation but the
bench box is GONE: "neurotic" unresolvable (Mac DNS=8.8.8.8, search domain "home" NXDOMAIN, mDNS
dead for ALL local hosts), full 192.168.4-7.0/22 sweep finds NO Proxmox :8006, the only LAN sshd
(192.168.7.239, OpenSSH 7.9/Debian10) has a DIFFERENT host key than known_hosts' neurotic. Worked
hours earlier same day => box down/rebooted or network changed. USER ACTION NEEDED. Instrument
ready to run: Gantt per-chunk/per-worker (GZIPPY_CHUNK_PHASE=1, chunk_fetcher.rs +290 lines),
residual split into 5 named AtomicU64 subphases (gzip_chunk.rs +84), exact resume sequence in the
worker report. BANKED FROM VENDOR SOURCE (GzipChunkFetcher.hpp:141-186, ChunkData.hpp:151-176):
ALL rg --verbose duration fields are THREAD-SUMMED CPU (per-chunk Statistics::merge under mutex) —
validates SUM-vs-SUM comparisons; decodeDuration=custom bootstrap, decodeDurationInflateWrapper +
decodeDurationIsal = ISA-L path, Theoretical Optimal = (decode+wrapper+isal+append+applyWindow+
computeChecksum)/nThreads. ASYMMETRY CAUGHT: rg's Theoretical-Optimal numerator INCLUDES
applyWindow + CRC32; gzippy's pool_efficiency() (statistics.rs) EXCLUDES them — cross-tool
efficiency comparisons must add those terms to gzippy's numerator. GATE verdict provisional-YES
(gzippy decode CPU 1497 <= rg ~1781) pending a fresh same-session --verbose run when the box
returns. MEANWHILE: Fulcrum `locate` v1 build dispatched (advisor-specified scope: critical-path
extractor over GZIPPY_TIMELINE traces + closed wall ledger w/ residual + gated exemption probe
design) — buildable offline with synthetic-trace selftests.
## PHASE DECOMP + REFRAME: gzippy decode CPU ALREADY <= rg on silesia T4 — the gap is PARALLEL EFFICIENCY (81% vs ~92%) [2026-06-11]
Worker-phase probe (probe/worker-phase, /tmp/gz-wphase, DUAL-SHA pass). PHASE TABLES (held
PROVISIONAL — residual 14% > 5% conservation flag): silesia T4 SUM 1763.7ms = header 19.5 / marker
body 901.6 [proven-slack control] / ISA-L FFI 595.8 (33.8%) / apply_window 91.0 / RESIDUAL 246.8
(CRC32-over-ISA-L-output, boundary replay, finalize, setup, reserve). Tail chunk 174ms is
MARKER-ONLY (98% body). bignasa T8: body 95.6% slack, ISA-L 1.6%, residual 0.6% — NO actionable
phase; 0.986 = NEAR-BAR not TIE (binding bar memory), revisit only if silesia mechanism generalizes.
PERTURBATION (clean: sleep-primary, self-test 1.00+/-3%, linear 0.49 ratio, N=9 sha-OK): ISA-L FFI
phase IS on the critical path (+10.6%/+21.7% at +50/+100%). RULE-3 CAP + ADVISOR REFRAME (banked as
STRONG HYPOTHESIS gating next dispatch): speed-up ceiling ~0 — gzippy inflate-only SUM 1497.4ms vs
rg 1781ms = 0.84x (gzippy does LESS inflate CPU than rg!); ideal wall 441 vs actual 543 => 81%
parallel efficiency vs rg ~92%; non-ideal overhead 102ms vs 38ms, DIFFERENCE ~64ms ~= the 67ms gap.
THE GAP IS SCHEDULING EFFICIENCY / IMBALANCE, NOT ENGINE THROUGHPUT (caveat: rg --verbose SUM
aggregation semantics unverified — step 1 of next dispatch confirms or kills). ISA-L call
granularity VERIFIED FAITHFUL (inflate_wrapper.rs:31-43 = vendor isal.hpp:205-207 refillBuffer,
128KiB staging + 128KiB output segments BOTH tools; poison_reserved_tail is cfg(test)-only) — not
a lever. NEXT DISPATCH (advisor spec): (1) GATE: cross-tool per-byte decode-CPU normalization —
if gzippy <= rg within spread, engine/ISA-L work is DEAD for this cell; (2) parallel-efficiency
instrument: per-worker start/finish spread, straggler location, does the 174ms marker-only tail
chunk land on the critical tail (cross-ref confirmed_offset_prefetch_gap); (3) residual+apply
decomposition to <5% conservation; (4) sleep-perturb the straggler/top-residual — NOT ISA-L compute.
## SERIAL TAIL EXONERATED (LOW branch) — WORKER-SIDE time binds silesia T4; "route engine-W" rider STRUCK as premature [2026-06-11]
Serial-tail worker (no patches, frozen, sha-OK). silesia T4 isal this session: gz 543 / rg 476 =
0.877x (G=67ms, sigma=40ms) — the campaign now reads ratios as a BAND across sessions
(0.877-0.912 on identical protocol); ADVISOR-SET MEASUREMENT-INTEGRITY GATE: pass/fail at the 0.99
bar binds on the WORST session ([min,max] reported for transparency); inter-session drift (rg
absolutes 506<->476, 1021<->1063) is an uncontrolled box-state variable that must be understood
before ANY near-bar cell can be declared a PASS. DECOMPOSITION (3 traced runs, leaf self-time,
conservation PASS): consumer TRULY-SERIAL S = 13.37ms (writev 13.2; all else ~0) < sigma=40 =>
PRE-REGISTERED LOW BRANCH: serial tail NOT the binder; corrected-overlap oracle NOT authorized.
"Consumer idle-wait = 0" is an instrument artifact: try_take_prefetched_pumping
(chunk_fetcher.rs:1442) spin-pumps 283ms of SELF-time fully overlapped with workers — that IS the
wait => WORKER DECODE THROUGHPUT is the binder. ADVISOR STRIKE: the pre-registered "route
engine-W" action was a false dichotomy — on the ISAL build the marker body is proven SLACK
(MFAST_DISABLE flat) and the clean tail is ISA-L FFI (gzip_chunk.rs:156-176), so engine-W has NO
resident binding phase on this cell (consistent w/ banked "real-ISA-L => low-T deficit largely
NON-engine"); engine-W stays funded for the NATIVE build only. bignasa T8 HARDENED: N=17, 0.986x
(up from 0.975 provisional) — provisional LOSS row; right-skew unimodal, no bimodality; NOT a
separate lever target (fold into whatever the worker-side decomposition names; re-separate only if
its binder differs). NEXT DISPATCH (advisor-specified): worker-side PER-CHUNK PHASE decomposition
on BOTH cells — extend the gzip_chunk.rs:1092 instrument to self-time {bootstrap-header,
marker-body [control: expect slack], clean-flipped, clean-tail-ISA-L, apply_window, setup/alloc},
conservation-asserted, summed + longest-worker tail chunk isolated; compare per-phase vs rg
decodeBlock SUM (the discarded 1.33x SUM gap may be directionally real); then SLEEP-inject the top
non-slack phase (spin = 3x turbo artifact on this cell) — proportional wall response names the
binder; NO work-stretch before that confirmation.
## RE-MEASURE BANKED (advisor-gated): silesia T4 0.912x / bignasa T8 0.975x; 410ms = corpus confusion; publish-thread claim REFUTED [2026-06-11]
probe/remeasure-baselines (d87c205d). (1) FULLY-HONEST BASELINES (frozen, interleaved N>=9,
both-arms-file-sink, sha-OK, canonical masks) SUPERSEDE the matrix rows: silesia T4 isal gz ~556 vs
rg ~506 = 0.912x LOSS; bignasa T8 isal gz ~1094 vs rg ~1063 = 0.975x LOSS — IMPROVED from banked
0.918-0.940 (today's merges) but below bar; PROVISIONAL pending N>=15 (2.5% margin); rg absolute
drifted 1021->1063 between sessions (box-state flag; interleaved ratios stand). (2) The sched
worker's "bignasa ~410ms / CLOSED 1.018x" = CORPUS CONFUSION (a silesia number mislabeled bignasa,
likely native build); its own binary re-runs honestly at 957ms single-shot sha-OK — claim fully
retired. (3) GZIPPY_PUBLISH_TID_STATS (DUAL-SHA pass): publishes consumer/worker = 61/0 silesia T4,
51/0 bignasa T8 — the "11/17 worker-thread publishes" claim REFUTED; gzippy already faithful;
consumer-publish port MOOT. (4) Worker's "two drive() impls both fire stats" attribution is FALSE —
cfg prod/stub pair (parallel_sm vs not(parallel_sm)), mutually exclusive; note only.
NEXT (advisor-amended order): (a)+(c) ONE frozen session — FULL consumer serial-tail decomposition
on silesia T4 from existing trace_v2 LEAF spans (self-time, busy+idle==span; leaves:
window_publish_clean :1747, publish_windows :3994, dispatch_post_process :1898, block_finder_get
:1241, try_take_prefetched :1422, drain :3942, writev :4028, combine_crc :4115) with an
OVERLAPPED-vs-TRULY-SERIAL split (fulcrum flow), PLUS bignasa T8 N>=15 hardening. PRE-REGISTERED
(b)-rule in spread units: S(truly-serial) < sigma => serial tail not the binder, route engine-W, NO
oracle; S > max(sigma, 0.4*G) => authorize the corrected pipelined-drain overlap oracle (prereg
RESOLUTION #2, F1<=1.01x / F2>=1.05x / F3 between); else supervisor decides. (d) model 0.857 gets
its OWN decomposition later (model-specific dispatch-silence, chunk_fetcher.rs:1515 — do not presume
shared mechanism). Engine-W remains the funded separate track.
## SCHED-PHASE0 GATED — headline DISCARDED at vendor source; HoL RETIRED; consumer publish+dispatch partially causal [2026-06-11]
Worker (probe/sched-phase0, 5cfb334, /tmp/gz-sched0 — byte-transparent knobs only) reported a big
headline; advisor gate (verified in BOTH sources) split it:
DISCARDED: "silesia T4 primary = 88% spec-failure; rg 0%; fix GzipBlockFinder::get offset
resolution" — REFUTED: vendor GzipBlockFinder.hpp:121-137 ALSO returns partition-grid guesses for
speculative indices (header :26); rg speculates-with-markers like gzippy (banked 31.25% replaced-
marker symbols), so "rg 0% spec-fail" is architecturally false; the SPEC_FAIL_* counters
(chunk_fetcher.rs:3195-3200, inc at :3586-3605) count EOF tail-walk candidate rejections, NOT
wrong-start chunks (counter-semantics scar class again). The 1.33x decodeBlock SUM, if real, is the
funded engine-W symbol-rate track, not a block-finder defect. ALSO DISCARDED: "apply_window runs at
standard priority" — gzippy submits at -1 == vendor submitTaskWithHighPriority (BlockFetcher.hpp:
608-611 <-> chunk_fetcher.rs:2445,2455; both pools pop lowest key).
BANKED: (B) the ~40% HoL-stall finding is RETIRED at HEAD — 1 on-demand NOT_STARTED fetch/run;
mechanism = the BTYPE=01 prefilter deletion (block_finder.rs:376-387, 2026-06-09) removed the
phantom-decode source. (E, sign only) consumer-thread serial publish+dispatch is PARTIALLY causal
on silesia T4 (SLOW_PUBLISH 5ms -> +28ms via the 6 consumer-path publishes; SLOW_DISPATCH 5ms ->
+41ms; 4-core mask, magnitudes unsized). Hit-drive kill-switch flat on silesia T4 (slack).
RE-MEASURE (presumed measurement error, R1): worker's bignasa T8 "~410ms" + "CLOSED at 1.018x" —
irreconcilable with Phase-0's sha-verified 1085ms same day/box/file AND the banked 0.918-0.940x.
VERIFY-FIRST: "11/17 publishes on worker threads" (no thread-id instrument shown; all gzippy
publish sites are documented consumer-thread — needs a thread-id counter before any port).
NEXT (advisor-ordered): (1) supervisor fully-honest baseline re-measure silesia T4 + bignasa T8 vs
rg, canonical 8-core mask, both-arms-file-sink; (2) publish-thread-id instrument; (3) IF confirmed,
faithful consumer-publish port (vendor queueChunkForPostProcessing:557-575); (4) boundary-resolution
CEILING oracle (reframed F) before any block-finder work; (5) engine-W stays the funded separate track.
## PHASE-0 mfast PROBE: Row 1 FIRED — mfast THROUGHPUT lever REJECTED with mechanism (advisor-SOUND) [2026-06-11]
Probe branch probe/mfast-phase0 (10fda98b, merged for the re-runnable instrument). Knobs:
GZIPPY_MFAST_DISABLE / GZIPPY_SLOW_MFAST_MODE (localized in-'mfast inject — the global marker knob
GATES the loop off at :1911, so this is non-redundant) / GZIPPY_MFAST_PROF. Validation: V1 self-test
1.034, V2 OFF-vs-absent codegen 0.985 (wall-neutral), V3 DUAL-SHA all pass. THE DISCRIMINATOR
(bignasa-isal T8, frozen, interleaved N=7, all 42 runs sha-OK, arm0 1085ms spread 62ms):
MFAST_DISABLE (ALL marker decode -> careful loop at baseline cost = a ~1.69x decode slowdown) is
WALL-FLAT (-0.6%). Silesia-isal T4 same: +0.4%. Speedup ceiling for mfast ~ 0; cannot close the
7.5%/10% gaps. Scoping (advisor): rejects mfast THROUGHPUT only — a future per-critical-chunk
LATENCY hypothesis would need its own perturbation. Do NOT read the sub-spread spin slopes
(+2.6/+5.6%) as partial criticality — noise; arm1 carries the verdict. PROF: mfast = 0.500 of
marker-decode cycles. CAUTION for future probes: silesia-T4 spin-vs-sleep showed a 3x turbo
artifact — SLEEP is the primary slope estimator on that cell. CONTRADICTION RECONCILED (ledger
note): rung-d's "FLAT at f100" is a KNOB-UNITS mismatch, not a wrong number — f100 = fixed
per-event spin; arm5's +100% = region-proportional (arm5 same config = +17.9%); both straddle the
same slack knee, mutually consistent; f-knob != percent — label units when reusing. NEXT (advisor-
gated): scheduling-side locate->confirm->map brief; lead with the hit-drive kill-switch A/B (the
:1530 port postdates the banked ~40% HoL-stall finding — quantify what it already closed); wire to
scheduling-ceiling-prereg RESOLUTION #2 (corrected overlap oracle = the open decider); treat
bignasa T8 (flip=0, 100% marker) and silesia T4 (~2/3 ISA-L post-flip) as potentially DIFFERENT
divergences.
## ADVISOR GATE on the fast-path relocation: SOUND-WITH-CHANGES — honest prior LEANS AGAINST the lever [2026-06-11]
Opus advisor (verified marker_inflate.rs:1911 + :2524-2533 in source) accepted the Phase-0 probe but
REFRAMED the evidence: rung-d's injection forced fast->careful — i.e. a ~1.69x decode slowdown (the
gap the mfast loop was built to close, :1889) PLUS +100% spin — and the wall stayed FLAT. That is
AFFIRMATIVE causal evidence that AGGREGATE marker decode rate has slack at bignasa-isal T8; the
decomposition's "6% slower per chunk" is a producer-side attribution (the analyst-biasable class).
Quantitative crux: for fast-binds + careful-flat to coexist you'd need "careful overlaps better
despite being slower" — strained/unevidenced. The COMPETING binder is the BANKED confirmed-offset
HoL-stall finding (~40% of T8 wall, decode_NOT_STARTED at confirmed offsets != partition guess) which
the decomposition never reconciled against. 5 REQUIRED changes to Phase 0: (1) fast-OFF/no-spin
DISCRIMINATOR arm (GZIPPY_MFAST_DISABLE-style, careful at baseline cost) — wall-flat => STOP, decode
slack, no Phase 1; (2) one-session interleaved arms + binary-self-test (the 973-vs-1350 anomaly is a
rule-4 flag); (3) pre-registered quantitative slope falsifier, not "monotonic"; (4) Phase-0-FAIL
pre-committed as a rule-7-valid rejection routing to the HoL-stall lever; (5) engagement-counter
TIME breakdown (mfast vs careful-tail vs table-build) — the deficit could live in the careful TAIL
(lever = reduce tail engagement, not speed mfast). OFF-vs-absent codegen self-test required for the
new in-loop knob.
## SERIAL-RESIDUE DECOMPOSITION DONE — chain exonerated; lever RELOCATED to the marker FAST-PATH loop [2026-06-11]
The bignasa-isal T8 serial chain is NOT the binder — every stage causally cleared: (1) WRITER:
skip-writev removal oracle saves 92ms of pure write time but an overlap-writer A/B is a TIE — the
write is already hidden under parallel decode; (2) APPLY_WINDOW: runs 6-way parallel, hidden by
decode, not on the critical chain; (3) DRAIN: ~2ms; recv_us=0 on all chunks (consumer never waits
on the channel). RELOCATED LEVER: per-chunk MARKER DECODE RATE — gzippy's marker loop is ~6%
slower per chunk than rapidgzip's on the same chunks, and that per-chunk deficit IS the ~68ms
(7.5%) bignasa-isal T8 gap. CRITICAL CAVEAT on the rung-d "marker slack" refutation: that probe
injected into the CAREFUL loop only (forced careful), but ~69% of marker events take the FAST
path — the slack finding does NOT cover the fast path; both reads are consistent: careful loop
has slack, fast path is the binder. ANOMALY FLAGGED: this worker's frozen absolutes (T1 973 /
T8 909ms) did not reproduce the banked 1350/1086 — frequency-state difference suspected; ratios
and structure consistent; treat absolutes as run-local until reconciled. NEXT: port the
deprioritized N2-N5 P3 passes (local-Bits mirror, lit-chain, c1/c2/c4 schedule, hoists — all
proven in the clean contig loop) to the marker FAST loop, which has received none of them.
## RUNG-D INC-1 MERGED (advisor-PASS) — marker attribution refuted; SERIAL RESIDUE is the next target [2026-06-11]
Marker DistTable TIE-KEPT (one dist-decode shape engine-wide; 72/72 grid; trajectory-complete
differential incl. flip-arming). THE PROBE REFUTED the isal-decide attribution: marker loop has
>=2x slack under bignasa-isal T8 (flat at +100% inject; conservative — careful-loop-forced);
T1 marker injection vacuous. RELOCATED TARGET: the serial residue chain at high-T isal —
T1 1350 -> T8 1086 = 1.24x scaling (drain / apply-window / marker-replace / writer). NEXT:
causal decomposition of that chain (fulcrum trace under the canonical mask; perturb/remove the
chain stages — apply-window has the parallel machinery, is it engaged on isal? marker-replace
volume on bignasa = 100% of bytes...), then the lever. N2-N5 marker Rust passes deprioritized
(slack). Doc-hygiene notes from the gate applied to the eval doc in a follow-up if rung-d
continues.
## RUNG (d) EVAL + INCREMENT 1: TIE-KEEP — the bignasa marker-loop premise CAUSALLY REFUTED [2026-06-11]
Branch engine/asm-rung-d (worktree /tmp/gz-asmd, NOT merged). Phase 1 eval (plans/asm-rung-d-eval.md):
kernel-extend REJECTED w/ mechanism (contig-u8 contract + register envelope + asm-before-Rust-plateau
inverts the rung-c order); RECOMMENDED Rust-first N1-N5 ladder (the marker loop had ZERO P3 passes).
Increment 1 (N1, DistTable dist decode in the u16 mfast loop, commit 67431727): full gauntlet green
(947/0 local pure, 674/0 default, guest isal 939/2-isolation-passing-flakes, sha 72/72 across 4 corpora
x T{1,4,8} x 6 arms, differential w/ ground truth + engagement counters, GZIPPY_MARKER_DIST_TABLE=0
kill-switch + GZIPPY_MARKER_DIST_STATS effect probe EFFECT-VERIFIED: 11.59M backrefs swap arms on
bignasa-isal T8). FROZEN A/B: TIE everywhere (bignasa-isal T8 +1.1% 3/9; silesia-isal T4 -0.5% 5/9;
silesia-native T4 -1.9% 8/9 consistent-but-sub-bar). KEEP default-ON (rule 7a, byte-exact, unifies the
dist shape engine-wide). THE REAL FINDING (causal, not attribution): (1) T1 has NO marker decode at all
(inject hits=0 — window-absent chunks exist only under T>1 speculation); (2) bignasa-isal T8 marker
decode has >=2x SLACK under the wall (36.17M inject hits; wall FLAT at f100 careful+inject, only f400
moves it ~2x) — the isal-decide '166MB/s marker rate owns bignasa' read was the analyst-bias class the
process rules name. The bignasa-isal deficit lives in the SERIAL residue (frozen T1 1350 -> T8 1086 =
1.24x on 8 workers: drain/apply-window/markers-replace/writer). N2-N5 marker micro-levers DEPRIORITIZED
(slack-bounded); NEXT CAUSAL TARGET: the serial chain on bignasa-isal.
## FIRST ISAL DECIDE RUN — the marker engine owns the isal deficits too [2026-06-11]
fulcrum decide @ isal build (frozen, fingerprinted, banked): silesia T4 0.899 / T8 0.991
PASS-but-bimodal(N->99 owed) / T16 0.945 / model 0.858 / bignasa 0.940. ALL 20 knob A/Bs
CAUSAL-NULL (incl. the new isal_incremental_growth knob — ratio-reserve confirmed not the
bottleneck). THE STRUCTURAL READ: bignasa-isal has flip_to_clean=0, isal_chunks=1/22 — ISA-L
fully idle; 100% of bytes through the pure-Rust MARKER loop at 166MB/s. Everywhere else the
pre-flip bootstrap phase runs the same marker loop at 51-83MB/s. (COUNTER-SEMANTICS CAUTION
banked: body_bytes = bootstrap-phase bytes only; silesia isal DOES hand ~2/3 of total volume to
ISA-L post-flip — the worker's '2% ISA-L utilization' over-read; bignasa's 0-flip reading is
solid.) THE SHARED LEVER: the marker bootstrap loop is the UN-ASM'D sibling of the contig clean
loop — rung (c)'s kernel covers only clean contig. RUNG (d) CANDIDATE: extend the kernel
techniques (or a second kernel) to the marker-mode body loop — lifts BOTH builds (bignasa-isal
0.940 directly; T4-band pre-flip phases; model both builds; native window-absent chunks).
Tool hygiene items found: contig_prof banks need build-tagging (native-only rows fired
DIVERGES-FROM-BANK on isal — expected, annotate); DO-THIS-NEXT picked a vacuous engine row on
the isal build (adapter should suppress contig classes when engine idle).
## FULCRUM P2 MERGED — OSS-cut ready; 147/147 across 4 suites [2026-06-11]
All gate-ranked hardening + the verification gate's finishers (honest unkeyed-chain threat model;
non-empty supersede reasons; toy mixed-sink refusal). Fulcrum now: fingerprinted (8 compare fields
incl. comparator-version + host), ledgered (supersede/invalidate, pending-reconcile, hash-chained),
derived-not-self-reported environments, adapter-pluggable (SCHEMA.md + toy adapter proof),
dual-licensed, case-studied. REMAINING for the public cut: repo extraction, a second REAL adapter
(one of the user's other projects = the generality proof), CI, license/name = user's call.
gzippy continues with Fulcrum as the instrument: next decide run at this HEAD ranks the remaining
cells (isal model 0.857/bignasa 0.918/T4 0.905/T16 0.924; native T1 0.877/T4 0.870/model 0.727).
## FULCRUM PRODUCTIZED + MERGED (advisor SOUND-WITH-CHANGES) — the user's OSS directive, phase 1 [2026-06-11]
tools/fulcrum: standalone causal performance-decision engine; 8 scar-named invariants enforced
in-path (gate-traced); fingerprints; contradiction ledger; decision briefs with falsifiers;
105/105 self-tests; gzippy as the first adapter; live run reproduced the honest matrix + both
drift detectors fired on real data. P2 HARDENING (gate-ranked, queued): (P1-class) ledger
supersede/invalidate; comparator-version + host-identity fingerprint fields; (P2) derive sink/
freeze fields (no self-report); adapter-pluggable artifact loader (the dir schema is core-coupled);
brief robustness (perturbations key, name string-matches); (P3) LICENSE file, scar-text
genericization, fd hygiene, append-only documentation. THEN: repo extraction for the OSS cut.
## THE FULLY-HONEST MATRIX @ cb269664 — both arms file-sink; two attribution errors corrected [2026-06-11]
CORRECTION-OF-THE-CORRECTION: the interim 'corrected' table (T1 0.973/T8 1.075) was HALF-phantom —
rg re-based to file-sink but gz kept /dev/null numbers (the anchor worker's 'gzippy is
sink-insensitive' claim FALSIFIED: native pays ~110ms@T1 for real output too). ALSO the bar2
worker's 'regression from uncommitted mod.rs changes' attribution is UNSUPPORTED — the tree is
CLEAN (only its own runner script, removed). The new matrix (both arms regular-file, canonical,
frozen, sha-verified) is the FIRST fully-honest table:
ISAL 6/14 PASS (silesia T1 1.210!, T8 1.011; nasa T1 1.595, T16 1.104; ghcn 1.005; small 2.754).
NATIVE 3/14 PASS (nasa T16 1.132, ghcn 1.029, small 2.211); silesia T1 0.877 / T4 0.870 /
T8 0.966 / T16 0.973; model 0.727-0.763; bignasa 0.925.
ASM VALUE AT HONEST PROTOCOL: native silesia T1 1528ms (pre-campaign file-sink) -> 1050ms now =
-31% total; the kernel's relative -19.7% (same-sink A/B) stands. SINK LAW (now absolute): BOTH
arms SAME regular-file sink, ALWAYS — half-rebased tables are phantoms too.
REMAINING (native): silesia T1 -12.3%, T4 -13%, T8 -3.4%, T16 -2.7%, model -24-27%, bignasa -7.5%.
REMAINING (isal): silesia T4 -9.5%, T16 -7.6%, model -14%, bignasa -8%, nasa T4 -2.1%.
## RG-ANCHOR RESOLVED: sink-protocol artifact; CORRECTED BAR TABLE — NATIVE PASSES T8 [2026-06-11]
The 810-vs-917 rg discrepancy = the flip session's flip_3way.sh used /dev/null while the spine's
canonical rule is regular-file-on-/dev/shm (rg pays ~108ms@T1/~50ms@T8 real write cost there;
gzippy's streaming pipeline already hides its own — /dev/null denies gzippy its streaming
advantage; the spine's comment exists for exactly this phantom class). HISTORICAL ANCHORS VALID;
the canonical file protocol is the bar denominator (the wall users experience includes output).
CORRECTED BAR TABLE @ asm-on HEAD (rg=file-protocol, frozen): silesia T1 0.973 (2.7% from bar!) /
T4 0.957 / T8 1.075 WIN PASS / T16 ~1.14 WIN (extrapolated, banking owed) / bignasa T8 ~1.025 WIN
(extrapolated) / model T8 0.785. GZIPPY-NATIVE NOW PASSES silesia T8 — its first canonical-protocol
silesia pass — with T16+bignasa likely (bank them), T1 at -2.7%, T4 -4.3%, model the deep cell.
Other hypotheses ruled out with evidence (pycache 43-vs-15ms; uncore 0x82b stable; freeze eras).
NATIVE T1 HONEST TRAJECTORY: 0.578 (M6) -> 0.973 (asm-on, canonical protocol).
QUEUE: bank T16/bignasa file-protocol cells; full canonical bar matrix BOTH builds at this HEAD;
fix flip_3way-class scripts to the spine sink rule (instrument-registry note); Squishy refresh;
then the last levers: T1 -2.7%, T4 -4.3%, model 0.785.
## ASM DEFAULT-ON MERGED — all cells improve; RG ANCHOR RE-BASED (integrity correction) [2026-06-11]
The four flip preconditions landed + verified on real BMI2; kernel default-ON for production
native builds (kill-switch + BMI2 fallback intact; non-x86 untouched). Frozen 3-way: gz2 improves
ALL SIX cells vs asm-off, zero regressions (T1 1170->939, T4 545->508, T8 342->332, T16 276->266,
model 654->517, bignasa 904->899). 60/60 shas, suites 946/0.
INTEGRITY CORRECTION: live co-located rg T1 = 809-814ms (sha-verified, verify/no-verify ruled out
CRC), NOT the banked 926.6 — the '0.98x' claim was stale-anchored. HONEST CURRENT BAR TABLE
(rg/gz2): T16 0.992 PASS-adjacent, T8 0.973, bignasa 0.969, T1 0.863, T4 0.86, model 0.71.
QUEUED: (1) rg-anchor investigation (why 810 now vs 905-927 across all prior sessions — rg-side
state? prior-condition artifact? ALL historical rg-relative numbers suspect until resolved);
(2) full bar matrix both builds at this HEAD with live rg arms; (3) Squishy refresh; (4) the
remaining cells' levers with fulcrum decide on the new HEAD.
NATIVE TRAJECTORY (honest): M6 0.578-vs-stale-anchor -> TODAY 0.863-vs-live-anchor, with every
intermediate ratio needing the same re-base lens.
## ASM RUNG (c) MERGED DEFAULT-OFF (advisor-SOUND) — NATIVE T1 0.98x UNDER THE FEATURE [2026-06-11]
THE FUNDED CAMPAIGN'S PAYLOAD LANDED: full-symbol-loop asm kernel, byte-exact (30/30 shas, 27k
differential trials on real BMI2, 3 live positive controls), asm_frac 0.998, frozen T1 silesia
-19.2% (944 vs 1169ms) => ~17ms from rapidgzip (0.98x); model T8 -20%. ~35% of the decode ceiling
captured in one rung. The c3 contract discovery (multi-literal+trailing-length = 99.6% of
crossings) was the make-or-break. Feature asm-kernel default-OFF; production builds bit-for-bit
unaffected. FLIP PRECONDITIONS (gate): consume-off-by-one control; 25:-arm coverage assert;
asm-ON/OFF random-gzip fuzz net; the +44ms OFF-arm layout tax (hoist or document).
NATIVE T1 FULL TRAJECTORY: M6 0.578 -> slab 0.712 -> chain ~0.72 -> ASM(feature) ~0.98.
NEXT: the four flip preconditions, then the default-flip frozen re-measure, then the bar matrix
both builds with asm-on native (isal untouched by this path), then Squishy refresh.
## ASM PHASE OPENED — charter merged; increment (a) falsified; rung (c) is the shape [2026-06-11]
Charter (plans/asm-campaign.md): budget math + 3-rung ladder + gauntlet + VAR_VIII salvage verdict
(prototype banked at bench/var8-fullkernel @ 922d6cbe). Increment (a) micro-kernel: byte-exact,
frozen +16ms T1 REGRESSION, reverted — seam tax ~1.4cyc x 15.9M crossings; LLVM can't schedule
across asm! boundaries. The falsifier CONFIRMS the design premise: rung (c) full-symbol-loop asm
(one seam per fast-loop run) is the only shape that can spend the ~620ms decode-latency budget.
NEXT SESSION: rung (c) — the full-kernel port against the P3.1-P3.5 loop state (local-Bits mirror,
DistTable, lit-chain, preloads as the register contract), VAR_VIII salvage as the starting point,
the charter's gauntlet per increment. This is the campaign's final instrument for native T1/model.
## ASM PHASE OPENED — charter banked; VAR_VIII salvaged from an uncommitted worktree; increment (a) NO-SHIP w/ quantified boundary tax [2026-06-11]
Branch engine/asm-p1 (worktree /tmp/gz-asm). CHARTER plans/asm-campaign.md: ladder (a) litlen
asm! micro-step -> (b) fused litlen->dist -> (c) full kernel (entered only on (a)/(b) evidence);
budget math (decode ceiling 642.6ms/50.9% T1, entire ~235ms rg gap fits 2.7x over; ship bar >=2%);
per-increment gauntlet + asm-kernel feature policy (x86_64-only, GZIPPY_ASM_KERNEL=0, default-OFF).
VAR_VIII SALVAGE: prototype existed ONLY as uncommitted mods in .claude/worktrees/var8-fullkernel
— now BANKED (bench/var8-fullkernel@922d6cbe). Salvage: F1 gather + F3 refill asm (layouts
unchanged at HEAD), exit protocol, KernCtx, coverage gate. Rewrite: F2 dist (ISA-L small LUT ->
libdeflate DistEntry), D copy (-> P3.4 shape), arms (+P3.2 chain, +Q3 lit1). INCREMENT (a) BUILT
+ NO-SHIP (b5c3f7c4, kept default-OFF per rule 7a): byte-exact everywhere (asm==ref 400k states
on real BMI2; sha 12/12; kill-switch proven; 10.9M hits T1) but frozen n=9 interleaved says ON
is WORSE: T1 silesia +16ms (-1.3%) vs same-binary OFF, T8 model +14ms; OFF~=base. MECHANISM
BANKED (F-a): ~1.4 cyc/step asm-seam tax x 15.9M crossings, zero latency recovered — per-symbol
asm seams are pure cost on the NEW loop too (DIS-1 re-confirmed, 10x finer); a winning asm shape
must amortize the boundary to ~once-per-block (rung (c); VAR_VIII had ~4000x fewer crossings).
Local trap re-hit + dodged: Rosetta exposes NO BMI2 — local "asm tests pass" was ref-only; the
real asm differential ran on the guest. Disk 100%-full local: reused /tmp/gz-chain scratch target.
## P3.5 DECODE-CHAIN MERGED (advisor-SOUND) — Rust reordering near exhaustion; ASM MEMO COMPLETE [2026-06-11]
c1+c2+c4 shipped byte-exact (T1 frozen 1183->1162, +1.8% of wall = ~3.3% of the decode ceiling;
every prof class improved frozen); c3 NO-SHIP on hardware-counter mispredict evidence. THE ASM
MEMO now has its complete, gated data chain: (1) decode chain = 50.9%/47.1% of the engine-cell
walls (removal oracles, disjointness-proven, conservative); (2) store side exhausted (<=94ms,
P3.4 took it); (3) BMI2 instructions already emitted (disasm) — the cost is DEPENDENCY LATENCY;
(4) Rust-level rescheduling extracts only low-single-digit % of the ceiling (P3.5). CONCLUSION
FOR THE USER'S FUNDED CAMPAIGN: the remaining native single-core gap (~620ms-class at T1) is a
microarchitectural per-symbol-latency problem — the asm hot-loop port is the remaining
instrument, with a measured budget and the one-engine architecture ready to receive it.
NATIVE TRAJECTORY: M6 0.578 -> 0.712 (slab) -> chain +1.8% => ~0.72-0.73 class at T1.
Cleanup notes: my own orphaned pgrep-wait loops on the guest were mutually-self-sustaining
(pgrep -f matches sibling cmdlines) — killed; pattern documented here as a trap.
## REMOVAL ORACLES MERGED (advisor SOUND-WITH-CHANGES) — THE ASM-MEMO VERDICT IS IN [2026-06-11]
Frozen disjointness-proven ceilings on the native engine cells: STORE 94ms (7.4% of T1 wall) /
23ms (3.5% model-T8-masked) vs DECODE 642.6ms (50.9%) / 311.6ms (47.1%). The Huffman-decode/
bit-read DEPENDENCY CHAIN holds ~half the wall on both engine cells — 6.8-13.5x the store side,
which P3.4 exhausted (contig-clean scope). DECODE ceiling is a conservative LOWER bound.
THE FUNDED REWRITE'S REMAINING LEVER IS THE DECODE CHAIN: table-lookup + bit-extraction latency
chains (BMI2 instructions already emitted — the cost is the DEPENDENCES). Next-session arsenal,
in order: (1) decode-chain restructuring in Rust (deeper preload/speculation across symbols,
litlen+dist fused lookahead, branchless entry dispatch — libdeflate/ISA-L chain-shapes); (2) if
Rust exhausts: the asm hot-loop port with the ceiling as the budget. Gate trims banked in
plans/removal-oracle-ceilings.md (internal-ratio scope; rg-freeze cross-check owed; high-T
magnitude extrapolated). Instruments: GZIPPY_ORACLE_NOSTORE / record-replay, OFF-state
byte-exact, permanent.
## LOCALIZATION SPLIT BANKED — DECODE ~= STORE, both critical; whole-loop profile [2026-06-11]
Frozen 9-arm T1 silesia (sleep controls survived both knobs; Rules 1+2 satisfied; super-linear
slopes noted as cache-pressure compounding): STORE +526ms / DECODE +576ms at N=50 spin
(DECODE/STORE=1.095, ~8x noise but small). NEITHER sub-region dominates => the T1 symbol-rate gap
is WHOLE-LOOP, not concentrated — the profile where per-arm micro-fixes exhaust and the funded
campaign's remaining levers are: (a) the pre-registered REMOVAL ORACLES to size each ceiling
(emit_backref_contig prefilled for STORE; zero-decode oracle for DECODE — build both, the larger
ceiling = the lever); (b) the whole-loop asm rewrite decision with those ceilings as the data.
Knob calibration note: NS_PER_SPIN_ITER=0.32ns is arm64-calibrated; x86@1.4GHz actual ~1.05ns
(sleep/spin 0.51-0.62 explained); sleep signal unambiguous regardless. contig_prof cycle-shares
(backref 62.6%) do NOT translate to wall share — the perturbation is the load-bearing number
(another attribution!=wall lesson, instrument-confirmed).
QUEUE: removal oracles (next); oracle.sh env-dup fix; isal model/bignasa cells (the isal-side
band: ratio-reserve land helped, ISA-L-path levers unexplored since); Squishy full-matrix re-run
on current HEADs (the standing bar evidence refresh).
## BACKREF PERTURBATION (tool-queued action 2) — engine ON critical path; localization knobs found [2026-06-11]
Frozen 4-arm T1 silesia: sleep-control injection +0.678s (+53.6%, 100x noise) => the clean decode
loop IS the T1 critical path (Rule 1+2 satisfied; spin artifact confirmed sleep/spin=0.71 —
spin numbers void for magnitude). BACKREF-SPECIFIC verdict NOT measurable with GZIPPY_SLOW_MODE
(loop-wide: forces the careful loop; its ~0.54s switching overhead dominates — measures the fast
loop's structural benefit, not backref cost; Rule 3 properly applied: no ceiling claims).
NEXT LOCALIZATION (pre-registered): GZIPPY_SLOW_STORE=N (fires at literal-store + backref-copy on
the PRODUCTION fast path, marker_inflate.rs:2534/2583/2671/2752/2812) vs GZIPPY_SLOW_DECODE=N
(Huffman-compute) — the clean backref-vs-decode split; ceiling needs a removal oracle
(emit_backref_contig pre-filled), not slope extrapolation.
INSTRUMENT BUG FOUND (fix queued): scripts/bench/oracle.sh --kind perturb builds duplicate env
keys (GZIPPY_SLOW_MODE=50 GZIPPY_SLOW_MODE=1; env last-wins => ZERO injection) — the perturb path
is silently broken for SLOW_KNOB=GZIPPY_SLOW_MODE; arms were run via direct env. Also: knob hit
count 39.7M (contig+wrapper) vs contig-only 17.97M — covered-region accounting documented.
## SLAB T-CONDITIONAL MERGED (advisor SOUND+SHIP) — native T1 0.712, best recorded [2026-06-11]
The fulcrum-decide loop closed end-to-end: tool surfaced the lever -> iter-1 NO-SHIP (criteria
held; cap-neuter mechanism pinned) -> iter-2 CAUSAL-VERIFIED +108ms n=21 -> gate SOUND+SHIP ->
merged. Native T1 trajectory: M6 0.578 -> P3.1 0.66 -> P3.4 0.655-0.712 -> NOW 0.712 (1296.6ms vs
rg 926.6). T>2 zero-engagement (cache-residency regime untouched). RSS trade documented + gated.
NATIVE CELLS NOW (latest frozen): T1 0.712 / T4 0.812 / T8 0.948 / T16 0.969(UNRESOLVED band) /
model 0.633. Tool's remaining queued action: engine.backref perturbation (60.5% of classed
cycles, pre-registered GZIPPY_SLOW_MODE perturb). Embedder hardening items (non-blocking):
DECODE_THREADS reset on exit; warm-process dealloc mutex tax.
## SLAB RECONCILIATION ITER-1: NO-SHIP (criteria held) + the contradiction pinned [2026-06-11]
Branch perf/slab-t-conditional @ 9bebcada (committed, UNMERGED): T-conditional gate + bytes-budget
implemented; suite 942/0; sha 16/16; effect-verified (kill-switch works). NO-SHIP: T8 RSS max-run
+16.8% and T8 wall +2.6% (unfrozen, borderline) — AND the headline T1 win VANISHED (+0.4ms
CAUSAL-NULL vs the uncapped slab's -99.9ms N=21). PINNED MECHANISM HYPOTHESIS for iter-2: the
16MiB effective_budget hard cap EXCLUDES the chunk-class blocks (T1 buffers are tens of MB) that
produced the win — the reconciliation neutered its own lever; and T8 RSS movement implies the
T-gate engaged above its intended K (read the diff's K + budget interplay first).
ITER-2 SPEC: gate strictly num_threads<=2; budget at low T must admit the chunk-class block
(budget ≈ max-block-size x workers, NOT a fixed 16MiB); T>2 = slab fully off (then T8 RSS/wall
criteria are untouched by construction); re-verify with decide.sh (the T1 knob row must return to
CAUSAL-VERIFIED ~-100ms with effect counters; T8 rows must show zero delta both metrics).
NOTE the instrument ALSO matters here: the earlier -99.9ms was measured uncapped on bin-p35; the
decide T1 base in THIS run was already faster (1416ms vs p35's 1402-1543 band) — keep same-binary
A/Bs as the only causal currency. The tool's second queued action (engine.backref perturbation)
remains open.
## FULCRUM DECIDE MERGED (user directive delivered; advisor-gated, live-verified) [2026-06-11]
One run -> ranked causal table. Proven end-to-end THREE times on the box: (1) first run reproduced
banked findings + surfaced slab_alloc PAYS ~100ms @ native T1 (N=21 RESOLVED — the lever an earlier
session reverted on wrong-placement evidence); (2) the sha-field regression its own hard-fail
caught (read-slurp; fixed + self-tested); (3) the effect-predicate non-exclusivity it caught on
ITSELF (line-presence -> slab counters; EFFECT-VERIFIED now live). RSS per arm on knob rows;
reverted-knob phrasing = reconcile-with-prior-gate. QUEUED BY THE TOOL: (a) slab@T1 reconciliation
(wall PAYS ~100ms, RSS to check, prior revert was T16-mask-conditioned — likely T-conditional
default); (b) engine.backref hypothesis w/ pre-registered perturbation (60.5% of classed cycles).
Guest: /root/bin-f2-native has the slab counters.
## P3.5 OFFICIAL MATRIX @ a9fe662c — arsenal trajectory banked [2026-06-10]
4 WIN / 6 TIE / 1 noise-REG vs baselines. NATIVE rg-ratios now: silesia T1 0.648 (was 0.578 at
M6 — arsenal moved it -116ms), T4 0.808, T8 0.959 (16ms from rg!), T16 0.956, model 0.610 (was
0.577), bignasa 0.915. ISAL stable (T1 1.20 PASS, others unchanged). CONTIG_PROF @ T8: backref
62.6% of cycles (34.9 cyc/iter), litchn 22.9%, wrapper=0 calls (contig = sole production path),
disttbl reuse 3/1768 (memcmp arm ~dead as predicted), bootstrap body 171MB/s with 98% of bytes
through the marker loop on silesia.
ASM-MEMO INPUTS: T8 native is 16ms from rg; T1 remains 1.54x (raw single-core symbol rate);
backref kernel + marker-loop rate are the two named残 structures.
## P3.4 MERGED (advisor-SOUND) — the copy-shape lever: T1 -87ms, T16 recovered [2026-06-10]
Backref copy ported to libdeflate shape (overlap-correct for dist>=8 burst+stride, hand-proven at
the gate; envelope 264 < 266 reservation): silesia T1 frozen 1462->1375 (0.941 NEW-OWNS); T16 back
to bar-native class (0.983; p25 at 325ms class); model TIE (layout-wobble exposed by same-binary
kill-switch instrument). DistTable amortized (T16 -2.9ms); copy prefetch +12ms T1. NO-SHIPs with
proof: BMI2 dispatch (native builds already emit BMI2 — disasm), pos-32 prefetch (worse).
NATIVE T1 TRAJECTORY (frozen, vs rg ~914-921): official 1528 (0.578) -> P3.1 1387 -> P3.4 1375
(~0.665). The single-core gap to rg is now ~1.5x = the remaining raw symbol-rate question; the
pure-Rust arsenal has more (backref-arm micro-delta, litpack interplay) but the ASM DECISION
approaches with real data. NEXT: P3.5 official-cells re-measure with p34 (all native cells +
isal regression checks) -> then the asm/portability decision memo for the user with the full
trajectory. Guest: /root/bin-p34-{native 08817e01, isal 17112a3a}.
## P3.4 INNER-LOOP PASS — backref copy is the headline (T1 -93ms); DistTable amortized; BMI2 dispatch refuted by asm [2026-06-11]
Branch engine/p34-inner-pass @ 02e6f962 (worktree /tmp/gz-p34; bins /root/bin-p34*-native).
THREE items, each separately committed + frozen-measured vs its predecessor bin:
1. a3401a58 DistTable amortization (SHIP): static fixed-dist table (OnceLock) + same-lens reuse
   + allocation-reusing rebuild + GZIPPY_DIST_AMORT=0 kill-switch. T16 -3..-4ms cross-binary AND
   -2.9ms same-binary ON/OFF (0.991) — recovers ~half the P3.3b 8.6ms; T1 1.0016 TIE; model
   same-binary 0.9984 TIE (the cross-binary +6-8ms there is LAYOUT, proven by the kill-switch
   instrument). Counters: silesia 1051 builds/2 reuses — the lens-memcmp arm ~never fires;
   the levers are alloc reuse + fixed-table sharing.
2. 181d7c25 libdeflate-shape backref copy (SHIP, the headline): 5-word burst for dist>=8 (ANY
   length incl. overlap — old shape per-byte'd dist>=8 overlaps), stride-dist trick for 2..7
   (old: per-byte), broadcast RLE. Envelope <= *pos+265 inside the existing 266 reservation.
   silesia T1 -93.3ms (1472.7->1379.5, 0.937, NEW-OWNS decisively); T16 -1.8ms; model TIE.
   Permanent differential: dist 1..258 x len x 8 alignments + canary envelope assert.
3. 02e6f962 long-match source prefetch len>40 (SHIP: T1 -12.4ms 0.9911 NEW-OWNS; T16/model TIE)
   + TWO NO-SHIPS WITH MECHANISM: (a) BMI2 PEXT/BZHI runtime dispatch — REFUTED BY ASM: the
   native-pinned build already emits BMI2 throughout the contig loop (16 shrx/46 shlx/8 bzhi,
   ZERO %cl shifts; PEXT has no non-contiguous-mask site); dispatch is provably instruction-
   identical on every measured cell. (b) wrapper-style pos-32 prefetch: T1 +5.6/T16 +3.5ms —
   the contig loop's recent-output line is already L1-hot from its own stores.
NET P3.4 (frozen 3-cell, p33 -> p34 final, interleaved sha-verified): silesia T1 1462.0->1375.2
(0.9406 NEW-OWNS, iqr 2-3ms — the engine cell's biggest single-session step of P3); silesia T16
332.7->326.9 (0.9826; p33 reproduced P3.3b's 332.9 and p34's p25=323.6 sits AT the bar-native
325.0 class — the P3.3b regression is recovered); model T8 1.0036 TIE. Suite 941/0 on guest;
sha grid 16/16 {silesia,model,bignasa,storedmix} x T{1,4,8,16}. Guest staged:
/root/bin-p34-{native,isal} (+.feature/.sha256). ANOMALY log:
one A/B ran 3x fast — bench-lock TTL lapse (watchdog thaw); caught by absolute-level sanity,
re-frozen + re-measured. model T8 carries +-1.5% BINARY-LAYOUT wobble across all P3.4 bins —
same-binary kill-switch A/B is the disproof instrument that separated it from behavior.
## P3.3b T16 TRIAGE — P3.1's DistTable owns the -3%; P3.2 clean [2026-06-10]
4-way frozen interleave n=15 (bar/m6/p31/p33): bar med 325.0 / m6 329.5 / p31 333.6 / p33 332.9.
P3.1 carries +8.6ms (distributions barely touch: bar p75 331.0 < p31 p25 332.2); P3.2 = +0.7ms TIE.
Formally UNRESOLVED under the strict both-IQR rule only because bar's distribution is BIMODAL
(IQR 9.6 vs delta 8.6) — the regression exists in the data. MECHANISM (inferred, unverified): at
T16 the per-block DistTable build cadence across 16 threads (smaller chunks => more blocks/sec)
+ second-table cache footprint outweighs the per-backref savings; at T1/model the trade nets
strongly positive (+15ms T1, +8.4% model). FIX SHAPE for P3.4 (not a revert): amortize/condition
the build — candidates: skip the DistTable when the block's remaining contig span is small (size
heuristic), cheap same-lengths reuse check, or a build-cost shave; verify at T16 n>=15 + T1 + model.
P3.4 SCOPE (one inner-loop pass): dist-build amortization + backref-arm polish (21.8 cyc/iter vs
wrapper-class) + BMI2 PEXT/BZHI dispatch + table prefetch — then the asm decision with full data.
## P3.3a OFFICIAL RE-MEASURE @ 2644f8be — arsenal lever confirmed on the cells [2026-06-10]
vs official baselines (sanity 7/7): model T8 native +8.4% REAL (rg-ratio 0.574->0.622; litchn owns
44.9% of classed cycles there vs 20-22% silesia — corpus-dependent payoff now MEASURED); silesia
T1 native +2.4% (0.607->0.622, N=15 spread 1%); T4/T8 native TIE; isal cells ~neutral (litchn=0
on ISA-L tails, deltas within 2-10% spreads). WATCH ITEM: silesia T16 native -3.0% (gz1/gz2 0.970,
~11ms, spread 4-7% — borderline; suspects: chain code-size/icache at all-core saturation) —
investigate in P3.3b before the next lever lands on top.
GAP MAP AFTER P3.1+P3.2: native T1 0.622 (rg 1.6x single-core — raw symbol rate; BMI2/asm
territory), T4 0.791, T8 0.927, T16 0.939-0.948, model 0.622; isal silesia T8 0.969-0.992
(bar-adjacent), model 0.855. NEXT (order): P3.3b T16 -3% triage; backref-arm polish (21.8 vs
wrapper-class target); BMI2 PEXT/BZHI + table prefetch; asm decision when the pure-Rust arsenal
is exhausted. Guest: /root/bin-p33-{native 0509ecd8, isal 078f5c86}.
## P3.2 MERGED — runtime lit chaining: model T8 -6.8%, T1 -1.6%, lits/iter 2.57 [2026-06-10]
The first arsenal lever pays: lit iterations halved (21.75M->10.85M silesia), chain fires 6.3M
times at 2.72 lits/iter; model (literal-heavy) gains most. Gate: SOUND, all six surfaces; optional
hardening noted (C_N_LITCHAIN>0 vacuity guard on the production net — fold into P3.3). REMAINING
measured headroom in the contig loop: backref 21.8 vs old 23.6 cyc/iter (narrowed without separate
work; wrapper-class target); litpack arm. NEXT P3.3 candidates (in order): (1) the model/T4-band
re-measure on the official frozen cells with p32 (how much of native model 0.577 / T4 0.784 did
P3.1+P3.2 close?); (2) backref-arm polish (the remaining ~10%-of-class delta); (3) BMI2
PEXT/BZHI dispatch + table prefetch (arsenal items, untouched); (4) asm only if cells still short.
Guest staged: /root/bin-p32-{native,isal}.
## P3.1 MERGED — T1 regression recovered, native beats pre-arc baseline [2026-06-10]
Bisect: M3's contig distance chain owned the -4%; DistTable single-lookup in the fast loop fixes
it (frozen T1 1387 vs baseline 1402; T4/T8 clean). Gate ran the missing table-level differential
itself (238+ sets x 200k patterns) — now a permanent test; lazy build adopted (no marker-mode
alloc). Profiler module landed (GZIPPY_CONTIG_PROF). P3.2 ARSENAL TARGETS (measured): backref
73 cyc/iter (wrapper 61.5) = 61% of classed cycles; lit chaining 21.7M iterations vs wrapper's
14.2M for identical 27.8M literals (multi-literal LUT packing underused in contig); litpack only
fires 1.23M times. Next: P3.2 = close the lit-chaining gap (multi-literal emission in the contig
fast loop) + the remaining backref delta; then model-corpus + T4-band re-measure; then asm if
still short of the cells (native T1 0.578-class, model 0.577).
## ENGINE M6 BANKED — structural arc wall-neutral; ONE engine ready for P3 [2026-06-10]
Frozen 3-way vs official baselines (sanity gates 6/6): native T4/T8/model TIE within spread
(gz1/gz2 0.985-0.998); isal regression-free (silesia T8 1.006 still PASS; model 0.859 ==
baseline). ONE FLAG: native T1 -4% vs baseline (1528->1593ms, outside 2% spread; seeded_block=16
is the active path) — P3 ITEM #1: profile the seeded-block T1 path (suspects: 32KiB dict-prefix
copy x16 chunks, fold-driver vs wrapper loop deltas), recover then exceed.
THE ARC'S VERDICT: M1-M6 delivered the STRUCTURE (one vendor-shaped engine, six divergences
resolved, -1778 LOC dead graph, every step byte-exact) at zero wall cost — exactly the
preparation the funded rewrite needed. The WALL gains now come from P3: the symbol-rate arsenal
on the one engine (multi-literal lookahead, BMI2 PEXT/BZHI dispatch, table prefetch, asm hot
loops — CLAUDE.md authorizes full reimplementation; prior falsifications non-binding; targets
native T1 0.578, model 0.577/0.859, the T4 band).
Guest production-candidate bins: /root/bin-m6-{native 1114b5c1, isal fef579bc}.
## ENGINE M5 MERGED (advisor-SOUND) — the dead graph is gone; structural phase COMPLETE [2026-06-10]
-1778 LOC: MarkerRing engine+fork, canonical family (dead-by-cfg proven), DIV-6 helper, orphans.
DIV LEDGER FINAL: DIV-1 closed (M3+M4), DIV-2 kept-by-design, DIV-3 closed (M5), DIV-4 closed
(M2), DIV-5 closed (M2b), DIV-6 closed (M5). gzippy-native = ONE Block engine, vendor-shaped,
40/40 sha, wall-neutral-to-slightly-faster through the whole arc. lib_api T1 test gap fixed.
Cleanup item (pre-existing): benches/{bootstrap_marker_overhead,inflate_block}.rs import the old
deflate_block::Block path — cargo bench broken independent of M5; fix in passing next session.
NEXT: M6 — the engine-cell measurement (masked + FROZEN canonical 3-way at the official baselines:
native silesia T1 0.609 / T4 0.790 / model 0.577 vs /root/bin-bar-native d288ef9c + rg), then P3:
the symbol-rate arsenal on the ONE engine (multi-literal, BMI2 dispatch, prefetch, asm — CLAUDE.md
authorizes full reimplementation; prior falsifications non-binding). Guest staged: bin-m5-{native,isal}.
## ENGINE M4 MERGED (advisor-SOUND) — DIV-1 COMPLETE: gzippy-native is a ONE-ENGINE decoder [2026-06-10]
until-exact on Block (labeled no-C-FFI deviation; contract proven file:line incl. the corrected
footer observable; member-final byte-aligned convention explicit + tested — the BFINAL scar paid
forward). 80/80 grid, multi-member exact, suites green, masked perf -0.7..-2.5% (faster).
unified::Inflate now off EVERY native production decode arm. Gate notes: deletion trap's negative
assertion backed by exact_route_defaults_to_block; optional hardening = short/mid-block error-
coordinate nets. Pre-existing cleanup item CONFIRMED unrelated: tests/lib_api.rs
classify_single_member_t1 lacks IsalSingleShot in its accepted set (fails on isal build since
0e57d8d9) — fix in M5.
LEDGER: DIV-1 CLOSED. Remaining: M5 (delete dead engines/graph: unified::Inflate native arm,
legacy MarkerRing, canonical loops, drain_clean_u8 DIV-3/DIV-6 + the lib_api test fix), M6
(masked+frozen engine-cell A/B vs the official baseline), then P3 (symbol-rate arsenal on the
ONE engine: multi-literal, BMI2, prefetch, asm — the native T1 0.609 / model 0.577 cells).
Guest staged: /root/bin-m4-{native ae0586ea, isal 3d181b38}.
## ENGINE M3 MERGED (advisor-SOUND) — DIV-1 part 1: seeded chunks on the ONE engine [2026-06-10]
gzippy-native seeded-inexact decodes on Block (16/16 at T1 silesia); unified::Inflate off that
path; empty-seed vendor semantics adopted; 80/80 sha grid; suites 921/928 zero fails; wall TIE
(layered per rule 7 — the T1 residual is SYMBOL RATE, P3's territory, now cleanly isolated on
ONE engine). Gate confirmed prefix accounting across every production consumer (M3 is the FIRST
production user of data_prefix_len) + one non-blocking advisory (clean-window oracle from-0 read,
measurement-only). Bonus: contig HEADROOM+1 latent fix; wrapper encoded_size_bits=0 hole documented.
LEDGER: DIV-1 part 1 CLOSED (inexact-seeded); remaining: M4 until-exact (pre-registered contract,
labeled deviation), M5 dead-engine deletions (DIV-3/DIV-6 + unified::Inflate off the graph
entirely), M6 masked+frozen engine-cell A/B, then P3 (the symbol-rate arsenal on the one engine).
## ENGINE M2b MERGED (advisor-SOUND) — stored early-flip ported, DIV-5 closed [2026-06-10]
54/54 byte-exact both kill-switch arms; storedmix +8.7%; silesia 1.000; purely additive diff
(913/0). Gate proved case-2 unreachability + case-1 marker-drop byte-safety from the arming
arithmetic. Worker also fixed a Bits bitbuf/bitsleft invariant in the new bulk reads (caught by
its own nets) and flagged STALE MEMORY: GZIPPY_FORCE_PARALLEL_SM no longer exists in src (parity
scripts still set it as a no-op — cleanup item). Map refinement: case 1 is width-independent.
DIVERGENCE LEDGER: DIV-2 kept (meets-or-beats), DIV-4 closed (M2), DIV-5 closed (M2b);
remaining: DIV-1 (M3 — THE payload step), DIV-3/DIV-6 (M5 deletions).
NEXT: M3 — seeded/until-exact chunks onto Block, unified::Inflate off the native graph
(GZIPPY_SEEDED_BLOCK kill-switch; M4 contract pre-registered in the gate amendments; masked A/B
native T1/T4/T8 = the engine cells; empty-seed vendor divergence resolves here too).
## ENGINE M2 MERGED (advisor-SOUND) — Block on WidthRing, DIV-4 closed [2026-06-10]
Mechanical migration landed: 911/0 guest suite, 30/30 byte-exact grid both builds, masked perf
TIE-or-better (silesia T4 0.203->0.195). The gauntlet caught + fixed an M1 skeleton contract bug
(32KiB-only flip conflate; vendor = FULL ring, deflate.hpp:1772, seam can reach RING_SIZE-258).
DIV-4 closed en passant (seam = u8 view, temp Vec deleted). Empty-seed vendor divergence recorded
for M3 (vendor flips Clean on empty window :1757; gzippy preserves historical no-op until callers
move onto Block). Guest bins /root/bin-m2-{native,isal} staged. NEXT: M2b (stored early-flip
DIV-5, own kill-switched step) then M3 (seeded chunks onto Block — the DIV-1 deletion, the
campaign's payload step; GZIPPY_SEEDED_BLOCK kill-switch; masked A/B native T1/T4/T8).
## ENGINE DESIGN GATED: SOUND-WITH-CHANGES — execution-ready [2026-06-10]
Opus gate verified the P1 headline + all 6 divergences FIRST-HAND (the campaign's vendor-misread
history made this the critical check): DIV-1 confirmed on the hot path (every window-resolved chunk
takes the second engine, chunk_fetcher.rs:2564); the apply-trace contradiction RECONCILED (vendor
applyWindow iterates ONLY dataWithMarkers; on silesia that's ~the whole chunk => the lever is
marker-loop symbol rate). Binding amendments appended to plans/engine-u8-design.md: DIV-5 split
out of M2 (own kill-switched step M2b); M4 relabeled a no-C-FFI-justified deviation with the
unified::Inflate contract pre-registered. THE CAMPAIGN IS EXECUTION-READY: next session runs
M2 (WidthRing member migration) -> M2b (stored early-flip) -> M3 (seeded chunks onto Block,
GZIPPY_SEEDED_BLOCK kill-switch, the DIV-1 deletion) with the full verification gauntlet per step.
## ENGINE P1 MERGED — the rewrite target is the GRAPH, not the loop [2026-06-10]
P1 (engine/u8-arch-p1 -> merged): first-hand vendor audit found the u8-flip-in-place INTERNALS
already converged since 2026-06-07 (Block flips u16->u8 in place, clean bulk u8-direct, storage
mirrors vendor). The funded rewrite's real targets (plans/engine-u8-map.md): DIV-1 native's SECOND
clean engine (unified::Inflate on seeded/until-exact paths; vendor seeds the SAME Block —
GzipChunk.hpp:456-458) — M-plan deletes it from the native graph; DIV-3 dead alternate engines;
DIV-4 seam temp copy; DIV-5 vendor stored-block early-flip cases. DIV-2 contig-direct KEPT
(meets-or-beats, one store pass vs vendor two). Design: plans/engine-u8-design.md (M1-M6, each
byte-exact-gated + kill-switched); WidthRing skeleton 12/12 tests, unwired. NEXT: gate the design,
then M1/M2 execution; P2 masked A/B on native T1/T4/T8 (the engine cells).
## USER DECISION: NATIVE ENGINE REWRITE FUNDED + N=21 close-band resolution [2026-06-10]
THE GATE IS RESOLVED (user): fund the native rewrite. Charter: plans/engine-campaign.md (u8-faithful
arch first per banked memories; asm arsenal after; rg decode-all-then-apply shape to evaluate).
N=21 frozen resolution: nasa T4 RESOLVED-PASS both builds (med/med 0.9998!); bignasa T8
RESOLVED-FAIL 0.951 both; silesia T16 RESOLVED-FAIL 0.861-0.863 — BUT native tied rg in 1/21 runs
(0.3165 vs 0.3148: the fast state EXISTS; scheduling reaches it 5% — the silesia-T16 lever is a
SCHEDULING-STATE question, not raw rate). rg distributions: bimodal on nasa T4 + silesia T16,
4-level quantized on bignasa (anomalies banked verbatim in the worker report).
## OFFICIAL BAR MATRIX @ b8e4fe58 (frozen, canonical, sha-verified; bins 247a6087/d288ef9c on guest as bin-bar-*) [2026-06-10]
| cell | rg/isal | rg/native | verdicts |
|---|---|---|---|
| silesia T1 | 1.199 | 0.609 | isal PASS / native FAIL(engine, user-gated) |
| silesia T4 | 0.906 | 0.790 | FAIL / FAIL |
| silesia T8 | 1.013 | 0.952 | isal PASS / native FAIL |
| silesia T16 | 0.924 | 0.981 | FAIL / FAIL(close) |
| model T8 | 0.870 | 0.577 | FAIL / FAIL |
| model T16 | 0.864 | 0.588 | FAIL / FAIL |
| bignasa T8 | 0.958 | 0.967 | FAIL(close) / FAIL(close) |
| nasa T4 | 0.971 | 0.935 | FAIL(close) / FAIL |
| nasa T16 | 1.127 | 1.142 | PASS / PASS — native's FIRST bar passes |
| ghcn T8 | 1.017 | 1.036 | PASS / PASS |
isal 4/10 PASS, native 2/10 (first ever). Hit-drive merged; suite 899/1-infra. The close-to-bar
band (0.92-0.97: bignasa, nasa T4, silesia T16, native silesia T16) suggests remaining shared
structure; the deep fails are model (engine shape) + low-T (T4) + native-T1 (engine, user-gated).
Superseded guest bins cleaned; production = /root/bin-bar-isal + /root/bin-bar-native.
## SUPERVISOR CORRECTION to the KEY-MISMATCH framing — DIS-19 tension [2026-06-10, IMPORTANT]
The previous entry's "fix (2): re-key prefetches => window-SEEDED clean decodes" is PROBABLY A
PHANTOM and must NOT be implemented as framed: banked DIS-19 says rapidgzip marker-decodes the
SAME ~34.5% byte fraction — rg's prefetched chunks are equally window-less (speculation WITHOUT
windows is the architecture, vendor tryToDecode decodes with markers from guessed offsets).
Re-keying gzippy's prefetches to wait for predecessor windows would SERIALIZE the pipeline =
unfaithful + likely slower. The causal tool's "KEY-MISMATCH 94-98%" is a true FACT about lookup
keys but a WRONG lever inference — rg has no such lookup succeeding either.
WHAT SURVIVES of the decomposition:
1. SCAN-OVERLAP divergence (REAL, vendor-anchored): gz consumer blocks on block-finder scanning
   (scan_candidate wall-critical 82-120ms; rg 0-27ms — rg overlaps scan with decode). FIRST lever
   next session; read vendor's alternating findNext/tryToDecode loop ordering vs gzippy's
   with_sync_boundary_search (the sync-ification comment at chunk_fetcher.rs:3618-3633 traded
   overlap away deliberately — re-examine THAT decision under the canonical mask).
2. The WINDOW-ABSENT MARKER-PATH ENGINE SHAPE: rg decodes marker chunks with its (fast) marker
   engine then ONE cache-friendly u8 LUT apply pass over the whole output; gzippy runs the
   bootstrap flip-hybrid (marker loop at 158-224MB/s + flip + ISA-L tail + small apply). On
   marker-heavy spans rg's shape wins. This is the ENGINE-W question in its true form — the
   user-gated decision now has its precise target: the window-absent bootstrap rate + the
   one-pass-apply architecture, NOT the clean-tail (which is already ISA-L/at-rate).
Next session order: scan-overlap (1) -> masked A/B; then present the engine shape decision to the
user with these numbers.
## MASKED PIPELINE DECOMPOSITION — the residual gap has TWO NAMED OWNERS [2026-06-10]
Masked (0,2,..,14) traced decomposition, bin-head-isal vs rg, model+silesia T8 (gz:rg 1.23x/1.09x
best-of-3): serial output NEGLIGIBLE (<0.1ms delta); the gap = (a) per-worker decode rate (~37ms/
worker on model: gz's bootstrap block_body 51-361ms + scan_candidate excess vs rg=0 block_body —
rg NEVER runs a flip-to-clean bootstrap loop: it marker-decodes the chunk then ONE cache-friendly
u8 LUT apply pass over the WHOLE output (rg apply SUM 964ms model/4320ms silesia, 38-100x ours,
parallel + cheap/byte, NOT its bottleneck)); (b) scheduling waste ~12ms (gz pool fill 79-87% vs
rg ~91%). TWO STRUCTURAL DIVERGENCES NAMED:
1. SCAN NOT OVERLAPPED: worker.scan_candidate is WALL-CRITICAL in gz (120ms model / 82ms silesia;
   consumer blocks on the block-finder scan) vs rg 0/27ms — rg overlaps scan with decode.
2. KEY-MISMATCH (the old confirmed-offset-prefetch-gap memory, caught in the act): 94-98% of
   chunks decode window-ABSENT with ZERO timing causes — the speculative prefetch looks up the
   PARTITION-SEED key while predecessors publish windows at BOUNDARY keys. Re-keying/re-issuing
   prefetches at confirmed offsets would convert marker-bootstrap chunks into window-SEEDED clean
   (ISA-L whole-chunk) decodes — the single highest-leverage faithful-port convergence remaining.
NEXT SESSION: implement (2) first (chunk_fetcher prefetch re-key at confirmed offsets, vendor
GzipChunkFetcher prefetch semantics), then (1); all A/Bs MASKED + canonical spine.
## PER-WORKER RETENTION REVERTED — canonical-mask regression + broken kill-switch [2026-06-10]
Under the spine mask (taskset 0,2,..,14) model T8: bin-head-isal (pre-retention) 0.17-0.18s/285MB;
retention build 0.24-0.25s/380MB (-29% wall, +90MB RSS); GZIPPY_PW_RETAIN=0 changes NOTHING (wall
AND RSS identical => the PwHeader over-allocation/trim path runs even when disabled — the gate is
broken). Independently confirmed by the canonical frozen re-measure sanity-gate FAIL (gz1@HEAD 608ms
vs banked 466-498, rg exactly on banked). The retention worker's own A/B used FREE-PLACEMENT
conditions (SMT roulette baseline 0.22s) where it showed +10%; under physical-core placement the
baseline is 0.17s and retention destroys it. REVERTED from HEAD. Lesson banked: condition-match
A/Bs to the CANONICAL MASK before accepting allocator levers; verify kill-switches actually
disable the whole path.
PIN KNOB verdict (canonical re-measure): redundant under the spine mask (the harness already pins
physical cores at T4/T8 — TIE all 5 cells); REAL +23% for unmasked real-world invocations =>
worth merging as a UX win (default-policy decision pending) but NOT a bar lever.
## PIN-WORKERS KNOB: +15-29% same-loop frozen on every T4/T8 cell (UNMERGED; canonical re-measure owed) [2026-06-10]
Branch perf/pin-workers-knob @ 234285db (worktree removed; /root/bin-pin kept): GZIPPY_PIN_WORKERS=1
pins decode workers round-robin to distinct PHYSICAL cores (sysfs thread_siblings_list; T>phys-count
guard => no-op; non-Linux no-op; default OFF). Frozen same-loop A/B: silesia T4 +29.2%, model T8
+20.9%, silesia T8 +20.2%, bignasa T8 +14.8%, T16 TIE (guard). Suite 904/1-preexisting.
CAVEAT (do NOT bank the bar ratios from this run): the worker REWROTE the bench loop inline (spine's
process-substitution hung for it); its absolute gz1 baselines are 30-60% slower than canonical-spine
banked cells => only the RELATIVE pin deltas are valid. NEXT SESSION, IN ORDER: (1) canonical-spine
frozen re-measure of the pin knob (gz1=HEAD unset vs gz2=HEAD pinned via GZ2_EXTRA_ENV, now in
_parity_guest.sh); if the deltas hold, the T4/T8 band cells likely CROSS or approach the 0.99 bar;
(2) advisor-gate the default-ON policy (vendor does NOT pin — justify as converging rg's EFFECTIVE
1-thread/core spread under the same scheduler; heterogeneous/E-core topology care; consumer/post
threads unpinned); (3) merge; (4) re-run the full silesia+squishy bar matrix both builds (native
benefits identically — same pool); (5) then the residual ~22% pipeline-vs-rg gap at pinned placement
(1527 vs 1944 MB/s) is the LAST structural term before the engine question.
## SMT PLACEMENT IS THE RESIDUAL — and a PIPELINE PLACEMENT LEVER FOUND [2026-06-10]
Placement probe (bare FFI, 8 spans, masks): A(0-15 free)=2189 agg w/ EXACTLY 2 slow threads/run
(SMT co-schedule roulette); B(one/physical core)=2766 agg (357/worker — ABOVE rg's 1989-2102);
C(forced SMT pairs)=-27%/thread; D(4 phys)=B/2 exactly => COMPUTE/PORT-bound, NOT DRAM/L3.
PIPELINE under same masks: gzippy A=1006-1216 -> B=1527-1555 MB/s (+30-50%!) while rg A=1989 ~=
B=1944 — rg's runtime achieves near-1/core spread under the same scheduler; gzippy's doesn't
(8 workers + consumer + post threads co-land on SMT siblings). REMAINING pipeline-vs-rg gap at
B-pinning ~22% (1527 vs 1944) = the true pipeline overhead, now isolated from engine+SMT+alloc.
LEVER QUEUED: GZIPPY_PIN_WORKERS knob — round-robin workers onto distinct physical cores within
the allowed mask; frozen A/B; if cells jump, advisor-gate default-ON policy (heterogeneous-CPU
care: E-cores excluded here; production default needs topology detection).
## PER-WORKER RETENTION MERGED, DEFAULT ON (81690afb, advisor-SOUND) [2026-06-10]
The mid-T allocation lever landed: lock-free one-slot-per-worker huge-block retention + 4MiB
MADV_DONTNEED trim budget (VMA kept, residency capped — the trick that passes BOTH wall and RSS
criteria; uncapped retention failed silesia RSS +52%). Criteria PASS: model T8 +10% wall/-34%
faults, worst RSS +5%; frozen co-located +6.9% on model vs prior HEAD; 907/0 guest suite; 135
frozen sha-verified runs. Design fenced by this session's falsifiers (global budgets dead at every
size; mutex pool dead; segmentation dead). Gate verdict SOUND across 7 surfaces (swap-only
transfers => no ABA; header-below-user safe; dispatch invariant: huge blocks can never reach the
small-class free path).
Frozen scorecard with the lever (model T8 rg/gz 0.878, silesia T4 0.890, T8 0.963, T16 0.925,
bignasa T8 0.92-class): band cells all improved again but bar still unmet. REMAINING: (1) ~11%
BARE-FFI-vs-rg concurrency residual (bus/L3 class — characterize); (2) low-T lifecycle share of
silesia T4; (3) native T1 engine decision (USER-GATED). Pre-existing notes: vmsplice test hang on
guest (environmental); T>64 worker panic pre-exists (MAX_WORKERS=64).
## GLOBAL-BUDGET SLAB FALSIFIED AT EVERY BUDGET — fork (b) shape pinned [2026-06-10]
Budget sweep B in {8,16,32}MiB (unfrozen grid, sha-verified): B8 passes RSS but is a NO-OP (one
model chunk = 12-25MiB > budget => free-list never hits); B16/B32 bloat silesia RSS +24%/+15%.
There is NO budget window between useful and bloated for a PROCESS-GLOBAL byte budget. Slab stays
opt-in; no commit. => FORK (b) FINAL SHAPE: PER-WORKER retention of EXACTLY ONE huge block, sized
to that worker's last chunk (retained == live working set => RSS-neutral BY CONSTRUCTION; cache
hit rate == chunk cadence). Implementation seam: rpmalloc_alloc.rs slab keyed per-thread (or the
defer_chunk_recycle/return path with thread-affine reclaim). FIRST task of next session: implement
+ frozen-verify (expected: the slab's +13% model / -72% pgfault class of win, zero RSS cost).
Branch perf/slab-retention-policy (a1ecc834) remains unmerged (bytes-budget policy improvement,
default-off) — merge alongside or fold into the per-worker change.
## ALLOCATION-CLASS FORK RESOLVED: (c) DEAD, (a) DEAD, (b) survives [2026-06-10]
(c) rpmalloc config: DEAD with mechanism — huge-class (>3.94MiB) frees call _rpmalloc_unmap
UNCONDITIONALLY (rpmalloc.c:2470-2491); no cache feature reaches them (unlimited_cache A/B: TIE,
pgfaults 140K==140K); rpmalloc-sys 0.2.3 comments out rpmalloc_initialize_config (span size locked).
(a) segment SegmentedU8: DEAD — 3 LOAD-BEARING-HARD consumers (ISA-L FFI contiguous sink;
decoded_range &[u8] borrows; contig_decode_window raw-ptr backref decode), PLUS the prior measured
loss (segmented clean = 3.26x dTLB-walks / 1.42x cycles, recorded in segmented_buffer.rs:1-30),
PLUS vendor stores CLEAN data contiguously (DecodedData.hpp:278-281) — the 128KiB ALLOCATION_CHUNK
segments are the marker-path/append granularity, not the clean store. Full constraint table in the
2026-06-10 worker report.
(b) PER-WORKER BOUNDED REUSE: the survivor. Existence proof = GZIPPY_SLAB_ALLOC (-72% pgfaults,
+13% model-T8 wall; cost +20% RSS from PROCESS-wide retention). Design next session: retain ~1
buffer per worker sized to its last chunk (caps added RSS at ~the live working set), lock-free
cross-thread return (consumer frees → owner worker reclaims), or recycle via the existing
defer_chunk_recycle seam with the slab's size-keyed lookup. Frozen verification owed on land.
## SLAB RETENTION POLICY: implemented, default stays OFF (RSS criterion) [2026-06-10 early]
Branch perf/slab-retention-policy @ a1ecc834 (UNMERGED, worktree /tmp/gz-slab): bytes-budget
largest-first eviction (GZIPPY_SLAB_BUDGET_MIB, default 64MiB free-list cap vs old possible 576MiB),
GZIPPY_SLAB_ALLOC=0 now correctly disables. Budget sweep: model T8 +5-10% wall at 64MiB budget; but
silesia RSS +35% with slab on => default-ON criterion FAILED => stays opt-in. CAVEATS: the worker's
"frozen" table was NOT actually frozen (no lock acquired; LXC denied no_turbo write) — its rg ratios
are UNBANKABLE; only slab-vs-head same-session deltas are directional. NEXT SESSION ITEM: gate +
merge the branch (policy improvement, default-off, low risk) with a REAL frozen verification; ALSO
fix pre-existing test classify_single_member_t1 (expects ParallelSM at T1; IsalSingleShot since
0e57d8d9 — only fails under some feature sets).
The mid-T band lever now needs a DIFFERENT shape than process-level slab: per-worker retention or
rg-faithful thread-cache spans (rg keeps RSS LOWER than gzippy while reusing) — design next session.
## MID-T BAND DECOMPOSED (concurrent-oracle probe + two env falsifiers) [2026-06-10 early]
Isolated-oracle scaling probe (8 distinct model spans, no pipeline): ORACLE T8 agg 1292MB/s ==
pipeline T8 (~1268) => at T8 the pipeline structure adds ~NOTHING; the whole gap is per-chunk
decode+ChunkData lifecycle. BARE FFI T8 = 1879MB/s (per-worker 298 > rg's 263 — raw ISA-L is NOT
the problem); ORACLE vs BARE = 31% ChunkData-lifecycle overhead, page faults 1.25M vs 86K per sweep
(fresh SegmentedU8 huge allocations: rpmalloc munmaps >3.94MiB on free, refaults on next chunk —
rpmalloc_alloc.rs:30). Residual BARE-vs-rg ~11% at T8 = memory-bus/L3 class.
FALSIFIERS (bin-head-isal, unfrozen same-conditions x3):
- GZIPPY_MANUAL_BUFFER_POOL=1: WORSE (model 0.21->0.26s, RSS +80MB) — legacy mutex pool DEAD.
- GZIPPY_SLAB_ALLOC=1: model 0.21->0.19s (+10%, churn mechanism CONFIRMED) but RSS +200-300MB on
  silesia/bignasa for neutral wall — needs CAP/POLICY TUNING before production (per-worker-sized
  retention, or rg-faithful thread-cache config), not default-ON.
NEXT SESSION: tune slab retention policy (cap to ~per-worker chunk size), frozen A/B all band
cells, then the residual ~11% BARE gap (bus/L3) + native engine decision.
## RATIO-RESERVE BANKED (frozen, 59573be9 = /root/bin-head-isal da52c5d1): EVERY cell lifted; silesia T8 PASSES bar [2026-06-09 late night]
Frozen 3-way N=9 sha-verified: model T8 0.684->0.846 (+16pp); silesia T4 0.851->0.906; T16 0.915->0.939;
T8 0.949->0.993 PASS(>=0.99); bignasa T8 ~0.90 (19%/12% spread, TIE band). The ratio-informed initial
helps EVERY corpus (allocation pressure was corpus-general, worst on near-incompressible).
DAY TRAJECTORY (isal, frozen): silesia T1 0.90->1.19 WIN | T4 0.874->0.906 | T8 ->0.993 PASS | T16
0.914->0.939 | nasa T1 0.57->1.57 WIN | bignasa T8 0.79->0.90 | model T8 0.658->0.846.
STILL BELOW BAR: model 0.846, silesia T4 0.906, T16 0.939, bignasa ~0.90.
NEXT-LEVER CANDIDATE (unverified): "Buffer pool u8: hits=0 misses=0" on the clean-tail path — the
chunk output Vecs are NOT pooled (fresh alloc per chunk); rg reuses. Diagnose the pool wiring gap.
Guest binaries: PRODUCTION=/root/bin-head-isal (59573be9); /root/bin-post (65dbcff9) = prior;
bin-ld-*/bin-ratio deleted; /root/gzippy-head = source@HEAD (no target).
## MODEL-CELL MECHANISM FOUND: 8x UPFRONT RESERVE under 8-way concurrency (env-knob falsifier +41% wall) [2026-06-09 night]
Probe ladder exonerated FFI (~400MB/s all stop-configs), oracle accounting (~350MB/s full path
single-threaded), bootstrap (1.4ms/chunk). The 132MB/s/worker collapse appears ONLY concurrent =>
memory system. FALSIFIER (3/3 reproducible, unfrozen same-conditions A/B, /root/bin-post model T8):
GZIPPY_ISAL_INCREMENTAL_GROWTH=1 FACTOR=2: wall 0.31-0.34 -> 0.22s (+41%), maxRSS 390 -> 291MB.
Mechanism: 8x-compressed-span upfront reserve allocates ~6x actual output on near-incompressible
data (32MB reserve / 5MB output per chunk x8 workers => allocation+fault+dTLB pressure = DIS-14/17's
footprint thesis, now CAUSALLY tied to the worst cell). FIX DIRECTION (queued): ratio-informed
initial = clamp(span x ceil(ISIZE/compressed x 1.25), 4MiB, 64MiB) — model ~1.7x, ghcn ~9.8x (no
DIS-29 churn regression), growable remains the underestimate safety net. sha-verify owed in the A/B.
## LONE-DRAIN LEVER: FALSIFIED + REVERTED (wrong bytes at silesia T4; wall TIE everywhere it ran) [2026-06-09 night, SUPERVISOR]
The pair-drain-gate removal + trailing-subchunk END-window change (0f0f9fb6, advisor-gated SOUND,
899/899 native box tests, bignasa 9/9 byte-exact T2/4/8) produced DETERMINISTIC WRONG BYTES on
silesia T4 (CRC32 mismatch f9acf2fe!=a49e7187, exit 1, all 9 iterations) -> REVERTED (6e015b44).
TWO lessons banked:
1. CORRECTNESS: the interior-subchunk window semantics (window-at-START stored per subchunk) are
   load-bearing somewhere the gate's 7-surface analysis + the whole test suite missed; silesia@T4
   (multi-subchunk chunks at low T) is the discriminating shape. NO test covers it — OWED: a
   silesia-shaped multi-subchunk low-T byte-exact gate before any retry of this lever.
2. WALL: even where POST ran byte-exact, NO win materialized (model T8 gz1/gz2=0.994 TIE; native
   model 1.004 TIE; silesia T16 0.986; bignasa T8 0.970 slight-regress-in-noise). The model-trace
   "pair-drain gaps ~377ms" attribution did NOT convert at the wall — Rule 3 strikes again
   (attribution != verdict; the gaps overlap other critical-path work). Lever = DEAD unless a
   future trace shows otherwise on a corpus where it converts.
GUEST STATE: /root/bin-ld-isal + /root/bin-ld-native are the BROKEN 0f0f9fb6 binaries — DO NOT
measure with them; safe production binaries remain /root/bin-post (isal 65dbcff9 ba788312) and
/root/bin-native-post (03072e73 ff4615dd). Native-suite-on-box workflow validated (avx2 test
passes natively; 92s suite) — keep using it.
## MODEL-T8 DECOMPOSITION (post-phantom-fix) — not engine-W-full-stop either [2026-06-09 late, SUPERVISOR]
model T8 (isal-POST 0.677x): traced wall 827ms (untraced median 733; rg 410). Speculation CLEAN (48-50
accepts, 0 mismatches; phantom fix fires 1x). The gap: (a) TWO slow-decode outliers (277/294ms vs p50
104ms) whose PAIR-DRAIN (consumer needs chunk K+1 resolved to consume K) blocks ~377ms combined — at T4
zero outliers exist; (b) 2 timing-dependent speculative_missing stalls (+107ms fetcher_get anti-scaling
62->170ms); (c) marker bootstrap body_rate on model = 38 MB/s (vs bignasa ~300) — near-incompressible
data is ~10x slower PER BYTE in the u16 marker loop, though only 7.7MB total goes through it; flip
arms fine (flip_to_clean=51/51, isal_chunks=51). decode p50 104ms/chunk x51/8 workers = 663ms floor >
rg's whole 410ms wall => per-chunk decode RATE is the structural term on this corpus, AMPLIFIED by
pair-drain head-of-line. NEXT HYPOTHESES (unverified): outlier chunks = long window-absent prefixes at
38MB/s; pair-drain (K+1 gate) may be a divergence vs vendor (rg consumes K when K is ready?) — VENDOR
CHECK OWED before any change. gzippy-native bignasa lift banked: T8 0.698->0.894, T16 0.710->0.909
(same fix); native model 0.574 = worst native cell. /root/bin-native-post=ff4615dd kept; model.gz kept.
## PHANTOM-EOS REJECTION — the bignasa/compressible high-T stall mechanism FOUND + FIXED (+15.6%/+16.6% wall, box-banked) [2026-06-09, SUPERVISOR session, HEAD 65dbcff9]
THE CHAIN (each step box-verified, advisor-gated): bignasa T8 loss 0.79x traced NOT to engine-W but to
PHANTOM SPECULATION: window-absent seed-first decodes at 4MiB partition-grid bases parse random bytes as
BFINAL=1/BTYPE=01 (1-in-8/position), tiny-decode 23-495B to a phantom EOS, get ACCEPTED (vendor rejects these
structurally via footer+next-member-header validation, GzipChunk.hpp:481,491-498,626-645 + tryToDecode catch
:728-732 — gzippy lacked the validation), then discard at consume = ~150ms head-of-line re-decode x3-9/run.
- FIX 65dbcff9: (a) dynamic-only finder (BTYPE=01 prefilter DELETED, vendor DynamicHuffman.hpp parity —
  box-verified wall-NEUTRAL alone, layered per guardrails); (b) phantom-EOS rejection in
  try_speculative_decode_candidate (+ CM=8 hardening). Counter SPECULATIVE_PHANTOM_EOS_REJECTS.
- WALLS (frozen, interleaved N=9, 3-way co-located rg+PRE+POST, sha-verified): bignasa T8 1269->1098ms
  (rg ratio 0.799->0.923), T16 1165->998ms (0.821->0.958). Mismatches 3/4 -> 0/0. silesia T4/T8 within
  spread (no regression). STILL BELOW the 0.99 bar — but the largest single move of the campaign.
- LEDGER CORRECTIONS: DIS-29's "T8+ ISA-L ret=-1/-2 seeding bug" = MISDIAGNOSIS (never observed in 30
  instrumented runs; the residual fallback was the BFINAL until_exact 1-bit coordinate decline, fixed
  d9a4f996 advisor-SOUND). A probe worker's "symbol 286/287 LUT bug" = ARTIFACT of probing raw partition
  bases (count[8]-=2 is ISA-L's own igzip_inflate.c:322; marker LUT is faithful). The apply_window pass is
  6-way PARALLEL (fulcrum flat-SELF was cross-thread sum, not wall) — serialized-apply hypothesis DEAD.
- ALSO THIS SESSION: T1 single-shot route merged (silesia T1 1.19x WIN), >8x growable storm fix (nasa T1
  0.57->1.57), JOB-2 reserve fix, BFINAL exact-landing fix, all pushed; 3-way parity runner (--bin2) +
  _parity_guest.sh stats() printf fix shipped.
- OPEN (bar = >=0.99 EVERY T): isal silesia T4 0.874 / T16 0.914; bignasa T8 0.923 / T16 0.958; model T8
  0.658 PRE-phantom-fix (re-measure owed); gzippy-native ALL cells (shares the fix — re-measure owed).
  Guest binaries: /root/bin-isal=PRE ab8babe5, /root/bin-post=65dbcff9 ba788312, /root/bin-native=37feb342.

# Orchestrator status — NAMING TRUTH + TWO-PATH + 3-WAY FULCRUM mission

## DIS-29 [2026-06-09, owner/isal-incremental-growth @ 153da9d1, gzippy-isal bin 99d5758e] — the owed DIS-23 FORCE-REGROW byte-exact test + corpus-reframe-gate STORM FIX, on >8x compressible corpora that ACTUALLY force the regrow
VERDICT: GZIPPY_ISAL_INCREMENTAL_GROWTH fixes the >8x reserve-overflow storm at LOW-T, BYTE-EXACT, recovers T1 ~+20-30% — but it is a LOW-T-ONLY lever with a sub-8x REGRESSION, NOT a clean default-ON.
- BUILD TRAP CAUGHT: first build = stale warm-target b9eb0a73 (== DIS-28 no-growth bin), cargo declined to recompile the rsync'd 153da9d1; smoke test falsely showed "growth does nothing". Forced 37s recompile → true growth bin 99d5758e. (assert fresh Compiling + new bin_sha; never trust 0.18s Finished.)
- STORM MECHANISM (first-hand per-Ok(false)-site diag): T1 storm = 100% reserve-overflow (out_pos==cap, compressed_span×8 too small for 10x) — corpus-reframe-gate mechanism CORRECT at low-T. T8+ OFF fallbacks are MIXED: reserve-overflow + ISA-L ret=-1/-2 (INVALID_BLOCK/SYMBOL) after 78/367B on un-decodable speculative-seed chunks (present in OFF too, NOT a growth defect — separate seeding bug).
- STORM FIX (counters, default f4): T1 nasa 0/5→4/1, bignasa(820MB,20ch) 0/20→19/1, small 0/2→1/1; ghcn 7.77x (sub-8x control) 7/1→7/1 UNCHANGED (no storm below 8x). Residual T1 fb = window-absent bootstrap chunk.
- BYTE-EXACT (owed DIS-23 gate SATISFIED): every sweep arm sha=OK (OFF==ON==REF). Forced-regrow (f1, grow=1MiB): silesia 14ch fb=0 byte-exact (~12 regrows/chunk), nasa byte-exact. Grow-across-realloc IS byte-exact on the real >8x path silesia never exercised.
- WALL (frozen min-of-5, rg/gz): nasa T1 0.554→0.662 (+19.5%, 0.92→0.77s), bignasa T1 0.439→0.572 (+30%, 3.71→2.85s); T8/T16 OFF≈ON for ALL (bignasa T8 0.893→0.893) => storm does NOT tax parallel-T even at 20ch≫16T (high-T chunks are finished_no_flip clean, bypass the storm-prone oracle). ghcn 7.77x T1 0.941→0.800 (−18% REGRESS: f4 initial under-covers 7.77x → regrow churn with no storm to fix).
- RECOMMENDATION for Opus gate: KEEP gated (byte-exact, fixes >8x low-T storm + DIS-23 dTLB win). Do NOT flip CURRENT always-small-f4 knob default-ON (ghcn −18%, ~0 parallel-T gain). PREFERRED: RETRY-ON-None-WITH-GROWTH (keep 8x first attempt → no sub-8x regression; grow only on overflow → rescue >8x byte-exact). Strictly dominates both arms; owner-turnable. SEPARATE owed: the T8+ ISA-L seed-error chunks (un-decodable speculative window).
- Box released clean (RESTORE VERIFIED no_turbo=0, wd inactive), guest+neurotic pgrep-clean, leaked local timeout-ssh wrappers reaped (broken multi-word $SSH_JUMP via dotfiles timeout shim → acquired bench-lock.sh DIRECTLY). NEW tooling→Steward: scripts/bench/{storm.sh,_storm_build.sh,_storm_guest.sh}; WORKTREE-ONLY env-gated GZIPPY_STORM_DIAG instrumentation (STRIP before merge). OWES supervisor Opus gate. See plans/disproof-ledger.md DIS-29.

## DIS-28 [2026-06-09, owner/corpus-general] — CORPUS-GENERALITY cross-check (the owed DIS-24/25/26/27 Q5): engine-W ROOT is corpus-general, but the "T8-win/T16-loss" CONCLUSION is SILESIA-SPECIFIC and INVERTS across the ratio axis
The squishy-variety cross-check all four prior gates flagged "owed; silesia is rg's tuning corpus." New host-locked driver scripts/bench/{corpusgen.sh,_corpusgen_guest.sh,_corpusgen_build.sh} (parity/hicurve spine: env-scrub, frozen+quiet readback, regular-file sink, interleaved best-of-N=5, sha-OK EVERY run, per-cell path+counter capture; PINS the exact conclusion binary b9eb0a73 by bin_sha, NO rebuild). Production routing path-asserted; T16=T16-SMT (mask 0-15, container NEVER expanded). 4 squishy corpora (ratio 1.26x→10x, size 6MB→269MB) + silesia control.
**PRE-FINDING: the task's "T1→IsalSingleShot" premise is FALSE for this binary — `classify_gzip` routes EVERY single-member stream to ParallelSM at all T/size (no IsalSingleShot path exists here; that's owner/t1-singleshot-route). Near-incompressible `model` did NOT route to StoredParallel — it has real Huffman (51 chunks, flip_to_clean=49), refuting the gate's "near-incomp = stored = little Huffman = different" structurally.**
**THE CURVE (ratio=rg/gz, >1=gz wins, TIE=≥0.99):**
| corpus | T1 | T8 | T16 |
|--------|-----|-----|-----|
| silesia 3.11x (home-field) | 0.904 | **1.022 WIN** | **0.918 LOSS** |
| model 1.26x near-INCOMP | 0.888 | **0.685 LOSS** | **0.677 LOSS** |
| nasa 9.93x web-log | **0.566 LOSS** | **1.044 WIN** | **1.048 WIN** |
| ghcn 7.77x CSV | 0.945 | 1.009 TIE | 0.963 LOSS* |
| small 6MB 10x | **1.887 WIN** | **1.862 WIN** | **1.742 WIN** |
**READS:** (1) silesia REPRODUCES DIS-24 (T8≈1.02, T16-SMT 0.918≈0.912) → instrument validated. (2) The "T8-win/T16-loss" SHAPE does NOT generalize: on near-incompressible `model` gz loses ALL cells WORST at high-T (0.68 — engine-W exercised HARDER); on highly-compressible nasa/small gz WINS high-T. (3) Single-core engine-W deficit (T1 loss) is corpus-general in SIGN (0.57-0.95) but magnitude is dominated by a NEW corpus-general term — the FALLBACK STORM: on ≥~8x-compressible corpora isal_chunks=0, ALL chunks fall back (DIS-14 ~7.5x spike), crushing T1 nasa to 0.566. (4) Unifying mechanism: gz's deficit scales with COMPRESSED-SIZE/CHUNK-COUNT (model 213MB→51ch=worst 0.68; nasa 20MB→3ch=win) — the SAME DIS-24 chunk-count→engine-W binder, producing a high-T loss ONLY when compressed-size spawns many chunks. (5) Small inputs: gz WINS 1.7-1.9x (rg's ~64ms fixed startup >> gz's ~34ms).
**VERDICT: engine-W is the corpus-general ROOT (heaviest where marker-decode is heaviest = model); but DIS-24/25/26/27's "T8-win/T16-loss" is SILESIA-SPECIFIC — the high-T loss DEEPENS on near-incompressible and REVERSES to a WIN on highly-compressible/small. Real-world picture DIFFERS from the silesia headline: gz is a strong WIN on small+highly-compressible, TIE/marginal on mixed text, DECISIVE LOSS only on large near-incompressible — the INVERSE of a "stored=easy" intuition.** CAVEATS: N=5 (signs robust; nasa/ghcn high-T wins on wide 7-12% spread = TIE-or-WIN); T16-SMT not T16-Ephys (no E-core re-measure per corpus). Full record+table: disproof-ledger.md DIS-28. NEW tooling flag to Steward. NO Opus advisor in env => self-disproof + silesia control only; OWES supervisor Opus gate (NOT advisor-vetted). Clean: bench-lock RELEASED (RESTORE VERIFIED no_turbo=0), container nproc=16 throughout, local+guest+neurotic pgrep-clean, leaked timeout-wrappers reaped.

## DIS-27 [2026-06-09, owner/dis27-ecore-busy] — THE un-run discriminator: high-T E-core workers are BUSY (engine-W), NOT idle => (B) under-feeding REFUTED, OVERTURNS DIS-25
The cheap decisive measurement the B-vs-b-reconcile-gate (Q1/Q4) said was OWED before "only user decisions remain":
at the high-T LOSS cell T16-Ephys (8 P-core SMT threads + 8 E-cores, mask 0,2..14,16..23) split the per-core-type
WORKER busy/idle that DIS-26 never measured. Tools: mpstat -P per-CPU %busy + perf -a per-PMU (cpu_core=P / cpu_atom=E,
HARDWARE-EXACT on the i7-13700T hybrid) instructions+cycles. Instrument VALIDATED (Rule 4): frozen-idle baseline ~idle;
T8-Pphys control E-cores ~2% (out-of-mask => no leakage). DIS-24/25/26 binary b9eb0a73 (no rebuild, READ-ONLY harness),
frozen guest BENCH_LOCK=quiet, container LIVE-expanded 16->24 then RESTORED to 16 (taskset16 fails, VERIFIED), sha=OK
both tools, ParallelSM-asserted, 2 reproducible runs.
**RESULT: gz's E-cores are BUSY — 72% (run1 71.9 / run2 72.8), the BUSIEST resource (busier than gz's own P-cores 67%),
retiring ~1.59x MORE instructions than rg's E-cores (gz 4.5e10 vs rg 2.85e10) for IDENTICAL decoded bytes; gz E-IPC 2.34
> rg 2.13 (NOT memory-stalled).** rg's E-cores are LESS busy (53%). **VERDICT: ENGINE-BUSY (terminal, user-gated). (B)
under-feeding/work-distribution REFUTED — gz's E-cores are over-subscribed (the long pole), NOT starved.** This OVERTURNS
DIS-25's (B) verdict by direct worker measurement and VINDICATES DIS-26's "high-T DECODE-WAIT = the engine" at T16-Ephys.
**Reconciles DIS-25's 1.91x:** the ~3/4 "parallel-only, not single-core-rate" E-core extraction gap is NOT feeding — it is
the ~1.59x HEAVIER ENGINE-W amplified through the IN-ORDER pipeline (gz's E-core chunks carry 1.59x more work => each
takes longer on the low-IPC E-core => the slow E-core chunk becomes the long pole and head-of-line-stalls the in-order
consumer while BUSY decoding, capping gz's incremental E-core benefit at ~half rg's). Exactly the gate's
"engine-W-under-in-order-contention" reading, which it said "would still be the asm/clean-rate lever." Mechanism
(source): gz worker = pure-Rust u16 marker-decode (chunk_fetcher.rs flip_to_clean:855 + src/decompress/inflate/, DIS-18
~57% instrs); rg counterpart leaner Block::read (vendor/rapidgzip/src/rapidgzip/GzipChunkFetcher.hpp+ChunkData.hpp, fed by
src/core/{BlockFetcher,Prefetcher}.hpp). Converges B-vs-b onto engine-W-user-gated for high-T. STILL OWED (NOT this gate):
the per-chunk/ParallelSM T1 removal oracle (var8-gate 124ms) + squishy corpus cross-check (DIS-24/25/26 Q5). Full record +
tables: disproof-ledger.md DIS-27. NEW tooling scripts/bench/_dis27_ecore_busy_guest.sh (READ-ONLY) flag to Steward. NO
Opus advisor in env => self-disproof + validated controls only; OWES supervisor Opus gate (NOT advisor-vetted).

## DIS-25 [2026-06-09, owner/high-t-discriminate] — HIGH-T A/B GATE CLOSED: chunk-count binder DEAD, verdict = (B) heterogeneous work-distribution lever
**[SUPERSEDED by DIS-27 2026-06-09: the (B) under-feeding verdict below is OVERTURNED — direct T16-Ephys worker measurement
shows gz's E-cores are the BUSIEST resource (72%, NOT under-fed); the high-T loss is engine-W (1.59x heavier on E-cores),
user-gated. DIS-25's P1/P2/roofline data stand; only its (B) feeding INTERPRETATION is refuted.]**
Host-locked discriminator (scripts/bench/hitgate.sh + _hitgate_guest.sh; DIS-24 binary b9eb0a73, no rebuild, frozen,
best-of-7, sha-OK, ParallelSM-asserted; silesia). Guest config finding: LXC 199 is Proxmox `cores:16` => cpuset 0-15
(P-core SMT threads only, E-cores OUT); expanded live to `--cores 24` for the run, RESTORED to 16 (verified).
**P1 rg chunk count (FREE confirm):** rg `--verbose` Total Fetched GROWS with -P: 17(T4/8)->27(T9/10)->33(T12/14)->
44(T16)->66(T24/32) — NOT "~17 constant"; rg partitions MORE than gz (gz 14->34) yet wins => chunk-count over-partition
binder DEAD (doubly: gz partitions FEWER). **P2 per-core split:** gz 205.8/129.7 MB/s (P/E), rg 229.3/147.5; gz/rg
ratio 0.90(P)≈0.88(E) — engine-W deficit real but NOT E-amplified; E/P IPC deficit (~0.63) identical both tools.
**P3 roofline:** T8-Pphys gz WINS 1.026 (effGZ 0.350 > effRG 0.306) despite the 10% engine deficit; T16-Ephys gz LOSES
0.934. Adding 8 E-cores: gz +35ms (captures 5.9% of E roofline), rg +67ms (10.3%) — rg extracts ~2x more from the same
E-cores. **VERDICT = (B): a faithful high-T work-distribution lever** (gz under-feeds slow E-cores; in-order publish-
chain head-of-line-stalls on a slow-core chunk). NOT (A): engine-W is masked at homogeneous T8 (gz wins there) so it
is not the high-T-specific cause. rg counterpart = BlockFetcher prefetch cache + ParallelGzipReader work distribution +
deeper reorder window. Caveat: silesia-only (rg's tuning corpus). Full table: disproof-ledger.md DIS-25. NEW tooling
scripts/bench/{hitgate.sh,_hitgate_guest.sh} flag to Steward. NO Opus advisor in env => self-disproof only; OWES Opus gate.

## DIS-23 [2026-06-09, owner/isal-incremental-growth] — FAITHFUL INCREMENTAL OUTPUT GROWTH: dTLB half CLOSED (below rg), RSS half NOT (lazy-faulting)
The cache-locality / small-memory DESIGN drive (DIS-17 owed footprint falsifier). SOURCE-VERIFIED first-hand that
DIS-14's "rg feeds the WHOLE buffer (isal.hpp:258), same as gz" is WRONG: rg's `finishDecodeChunkWithInexactOffset`
(GzipChunk.hpp:309-379) allocates a FRESH 128 KiB `DecodedVector` per loop iter, fills via readStream, `resize`+
`append`s into a segmented DecodedData, repeats — it grows INCREMENTALLY in 128 KiB segments and NEVER reserves the
8x-compressed chunk output upfront like gzippy does. Built a gated faithful port (`GZIPPY_ISAL_INCREMENTAL_GROWTH=1`,
OFF==identity): small initial reserve + grow-on-demand inside the ISA-L loop (`decompress_deflate_from_bit_into_growable`
+ `IncrementalOutSink`). Growth DISSOLVES DIS-14's sub-8 fallback constraint (even FACTOR=1 is 0-fallback).
MEASURED (frozen guest, bin 908a9629, T4+T8, vs rg 0.16.0, full lib suite 893/0 OFF+ON, dual-sha byte-exact,
isal_chunks=14 fb=0 every arm): **dTLB-miss MPKI T4 0.0623->0.0366 / T8 0.0685->0.0396 (factor-4, -41/-42%, lands
BELOW rg 0.0466/0.0502)**; page-faults -8..11%; **wall TIE (no-regress)**; **peak RSS NOT improved** (f4≈OFF; f2/f1
WORSE). VERDICT SPLIT: the user's named-unmet TLB half is CLOSED (wall-neutral, 0-fb, byte-exact); the RSS +21-25%
half is NOT — lazy faulting means RSS tracks TOUCHED bytes + realloc transients, not the lazy over-reserve (this
REFUTES DIS-17's "the +25% RSS is mostly the D1 8x over-reserve"). KEEP gated (rule 7a — correct + helps dTLB);
recommend the Opus gate flip default to ON factor-4. Full record + table: disproof-ledger.md DIS-23. Branch
owner/isal-incremental-growth (uncommitted-then-committed on the worktree); NEW tooling scripts/bench/{incr_growth.sh,
_incr_growth_guest.sh} flag to Steward. OWES: supervisor Opus gate.

## DIS-21 [2026-06-09, owner/defrag-flat-convergence] — the de-frag FLAT-BUFFER lever is a PHANTOM (CLOSES the DIS-20 owed speed-up bound)
DIS-20 left OWED: the rule-3 flat-buffer REMOVAL oracle (does flattening the u16 marker output recover the
~70ms T4 gap?). RESOLVED on two grounds:
1. **SOURCE (first-hand vendor read, the rg-marker gate's owed `m_window16` check):** rg's `m_window16` is
   `std::array<uint16_t, 2*MAX_WINDOW_SIZE>` (deflate.hpp:805) — the SAME 65536-u16 modulo-by-AND RING as
   gzippy's `output_ring`, NOT a flat per-chunk buffer. `resolveBackreference` (:1369-1390) ≡ `emit_backref_ring`
   (same offset-modulo, same single-memcpy :1376, same marker scan). rg drains `m_window16`->`dataWithMarkers`
   via `lastBuffers`+`appendToEquallySizedChunks` (DecodedData.hpp:266-275) into 128KiB/64Ki-u16 SEGMENTS ≡
   gzippy `drain_to_output`->`SegmentedU16::push_slice`. **gzippy is ALREADY structurally convergent. There is
   no flat buffer in rg to converge to; building one would DIVERGE.** The "flat m_window16" premise (DIS-19/
   defrag-charter) was a MISREAD. (rule-7(a) rejection: rg does the de-frag the SAME way.)
2. **MEASURED (the one faithful byte-exact convergence available):** `GZIPPY_FLAT_BACKREF=1` swaps
   emit_backref_ring's non-overlap word-copy (a gzippy innovation over vendor) for rg's EXACT
   `copy_nonoverlapping(length)` memcpy. Interleaved OFF/ON/rg N=15, frozen guest runnable 1.00, byte-exact
   (OFF sha==ON sha==ref), isal_chunks=14/0, path=ParallelSM: OFF 552.4ms / ON 549.7ms => **delta -3ms (min),
   -1ms (med), within ±30ms spread = CLEAN TIE.** instr-count: ON cuts ~96M (~1.5%) yet wall flat = textbook
   rule-3 (instr win does NOT translate to wall; the copy is off-critical-path / mem-latency-hidden).
VERDICT: **LOW-CEILING; the de-frag flat-buffer port is NOT worth funding.** DIS-20 criticality stands (slope),
but the convergence SPEED-UP ceiling is ~0. The isal T4 gap is NOT the u16 output/backref fragmentation — it is
the marker-prefix Huffman SYMBOL-RATE half (asm-bounded, user-gated) and/or parallel scheduling (DIS-16/17).
Full raw + provenance in plans/disproof-ledger.md DIS-21. New tooling (flat-backref KIND, flat_ab_interleave.sh,
GZIPPY_FLAT_BACKREF knob) FLAG TO STEWARD.


## RG-MARKER-ATTR — apples-to-apples split of the +1.54e9 isal excess [2026-06-09, rg-marker-attr worktree, DIS-19] — COMPLETES the isal attribution
The OWED counterpart of DIS-18: perf-record rapidgzip's OWN marker decode (the campaign always ASSUMED
rg-marker ~= 0). The installed rapidgzip 0.16.0 .so is ALREADY symboled (BuildID 67cd8b7e, debug_info, not
stripped) — NO build needed. Bench-locked frozen guest T4 mask 0,2,4,6, REPS=6, sha=OK, rg 4.7078e9 instr:u
(reconfirms DIS-18 exactly). FINDINGS:
- **rg NEVER takes the u8-direct ISA-L path (decodeChunkWithInflateWrapper = 0 samples)** for streaming -d -c;
  it marker-decodes EVERY chunk via decodeChunkWithRapidgzip->Block<false>::read, using the SHARED igzip AVX2
  kernel (address-confirmed ..@37.end/..@42.end inside decode_huffman_code_block_stateless_04) as its Huffman
  primitive, then resolves via DecodedData::applyWindow. gzippy's flip_to_clean (12/14 -> real ISA-L FFI) is a
  DIVERGENCE from rg, not a convergence.
- **PER-AXIS split of the +1.54e9 excess: ~71% marker INNER LOOP (+1.09e9) / ~25% resolution-SCAFFOLD (+0.38e9);
  shared igzip kernel ~equal (gz 1.88e9 ~= rg 1.68e9).** gz inner-loop 3.338e9 vs rg 2.246e9; gz resolution
  0.715e9 vs rg 0.335e9. The pre-registered FALSIFIER resolves to the **INNER-LOOP branch** (open inner-loop-
  engine, VAR_VIII plateau ~0.667x PREDICTED) — NOT pure-scaffold-faithful-convergeable.
- Inner-loop excess decomposed: ~1.70e9 is the u16 OUTPUT/BACKREF ABSTRACTION (push_slice 0.706e9 SegmentedU16 +
  emit_backref_ring 0.993e9 u16-ring) that rg FUSES into inlined flat m_window16[pos++] writes — the largest
  tractable sub-lever (flatten+inline+fuse like rg's single Block::read; architectural). ~1.61e9 is pure-Rust
  Huffman where rg uses the ISA-L kernel — the inner-loop-asm question (plateau-predicted).
- BAR-1: closing isal T4 0.91x->0.99 by converging the marker engine alone is UNLIKELY to clear the bar.
  gzippy-NATIVE shares this marker engine + its 0.667x clean floor. NEW tooling (flag to Steward):
  scripts/bench/{rg_marker_attr.sh,_rg_marker_attr_guest.sh}. Full numbers: disproof-ledger.md DIS-19.

## LOCATE-+40% — per-symbol attribution of the DIS-17 instruction excess [2026-06-09, owner/locate-plus40, DIS-18]
perf record -e instructions on a SYMBOLED gzippy-isal (HEAD d56cb0f5; production codegen fat-LTO cgu=1 +
`-C strip=none -C debuginfo=2`, instruction count unchanged) DECISIVELY LOCATES the +40%: it is the
pure-Rust **u16-MARKER DECODE ENGINE (~57% of user instructions, ~3.56e9)** — `marker_inflate::Block::
read_internal_compressed` 19.6% + `emit_backref_ring` 15.9% + `SegmentedU16::push_slice` 11.3% +
`resolve_chunk_markers_on_chunk` 3.5% + `decode_chunk_unified_marker`/`marker_decode_step` 2.5% — NOT the
suspected wrapping. Frozen bench-locked guest, mask 0,2,4,6 (T4), cpu_core PMU, REPS=6, sha=OK, isal_chunks=14
fb=0 path=ParallelSM asserted. gz 6.252e9 instr:u vs rg 4.708e9 = **+32.7% user** (~1.54e9; DIS-17's +40%
total = this + the kernel/page-fault footprint); wall gz 0.531 / rg 0.470 = **0.885x**.
REFUTED suspects: u16->u8 narrow <0.01% (already fused in-place, converged), per-chunk CRC 0.78%, scaffold
~9.5% (finalize_with_deflate 7.9%), consumer scans/window-map <0.1%. ISA-L decode kernel ~30% is SHARED w/ rg.
Falsifier PASSES: the marker engine owns the excess — gzippy decodes 12/14 chunk PREFIXES through u16 markers
(window-absent at speculative decode start, flip_to_clean=12) where rapidgzip decodes window-PRESENT chunks
u8-DIRECT via ISA-L. The engine MODEL is faithful (m_window16 ring/set_initial_window); the divergence is the
marker-decoded FRACTION. CONVERGENCE (ranked, supervisor-gated, NOT attempted — architectural not byte-
transparent): (1) decode window-present chunks u8-direct via ISA-L from the start (vendor GzipChunk.hpp
decodeChunkWithInflateWrapper) to shrink flip_to_clean toward rg's bootstrap-only == governing-memory "faithful
u8 native clean path" + confirmed-offset-prefetch-gap; (2) residual marker-engine per-symbol efficiency.
Tooling for Steward: scripts/bench/{perf_attr.sh,_perf_attr_guest.sh}. NOTE: guest target/release/gzippy is now
the symboled gzippy-isal (next --build rebuilds). Full raw in plans/disproof-ledger.md DIS-18. OWES Opus gate.

## LEAN-CONSUMER — T4 sizing checkpoint [2026-06-09, owner/t4-lean-consumer, DIS-16]
SIZED the T4 0.911x gzippy-ISAL parallel-pipeline lever the single-shot oracle (DIS-15)
could not probe. Built a BYTE-TRANSPARENT removal oracle `GZIPPY_LEAN_CONSUMER=1` that
skips the non-vendor consumer-top overheads (pipeline-fidelity-verdict.md D2 unconditional
per-iter process_ready_prefetches+harvest; D3 clean-branch full prefetch-cache scan; D4
throwaway format! trace Strings) + a `[LIFECYCLE]` GZIPPY_VERBOSE per-chunk self-time
decomposition. Measured ON-vs-OFF, frozen bench-locked guest (no_turbo=1, gov=performance,
runnable≈1.25, QUIET), gzippy-isal, interleaved N=13 ×2 runs, same-sink /dev/shm regular
file, vs rapidgzip 0.16.0; STEP-0 path=ParallelSM + isal_chunks=14/14 fb=0 asserted BOTH arms;
sha=028bd002…=OK every arm (lean byte-transparent T1+T4); bin 378788924ace0381;
raw_bytes=211,968,000.

VERDICT — NULL BRANCH (TIE): the faithful consumer-lean is NOT the T4 BAR-1 lever; the
T4 gap is DEEPER parallel-scheduling.
- T4 wall (mask 0,2,4,6): run1 gz-prod 0.5470s/0.902x · gz-lean 0.5524s/0.894x · rg 0.4936s;
  run2 gz-prod 0.5517s/0.892x · gz-lean 0.5471s/0.899x · rg 0.4921s.
- lean RECOVERY = gz-prod−gz-lean = −5ms (run1) / +5ms (run2): sign FLIPS, |Δ|≈5ms ≪ the
  17–36% gzippy spreads (~90–200ms) ⇒ recovers ~0. (No work-displacement: gz-prod and gz-lean
  iter_sum/postproc are identical to within noise; nothing moved to another stage.)
- LIFECYCLE decomposition (the deliverable, why it's null): the removed D2/D3/D4 are µs-scale —
  prefetch_promote 1→0 µs, window_publish ~0.5ms total, D4 format! sub-µs — 2–3 orders below the
  ~55ms wall gap. Consumer self-time is dominated by fetcher_get (productive ISA-L decode-wait
  ≈115ms cumulative/17 iters) + postproc_dispatch (≈74ms), NEITHER touched by the lean.
- OWED next: the T4 ~0.90x gap lives in the productive parallel decode + post-process /
  scheduling path (head-of-line stalls, window-publish serialization) — the parallel-scheduling
  oracle DIS-15 flagged. gzippy-NATIVE untouched (separate 0.667x engine floor).
- Artifacts: worktree .claude/worktrees/t4-lean-consumer @ 1642cb58; chunk_fetcher.rs
  `lean_consumer_enabled`+`[LIFECYCLE]`; scripts/bench/{lean.sh,_lean_guest.sh}. Code KEPT
  (byte-transparent dev oracle, useful instrumentation; not merged — sizing turn). Box released
  clean, no orphans (guest+jump+local pgrep clean, freeze thawed).

## PIPELINE-FIDELITY AUDIT — lead-auditor checkpoint [2026-06-09, SOURCE-ONLY, READ-only on parallel/]
Compared gzippy coordination + serial tail first-hand vs ACTUAL rapidgzip headers. VERDICT in
`plans/pipeline-fidelity-verdict.md`. HEADLINE: the coordination/serial STRUCTURE is LARGELY
FAITHFUL — the hunch ("we introduced a coordination/serial deviation") is only WEAKLY borne out.
No single large divergence found; deviations are real but small/diffuse. KEY CORRECTION: the
"per-chunk FFI handoff" residual term is NOT a gzippy deviation — vendor's IsalInflateWrapper
(WITH_ISAL) pays the identical per-chunk setup, so it can't explain the gap vs rg. Ranked
deviations (vendor↔gzippy file:line + runtime work + converge target in the verdict): D1 ISA-L
output over-reserve 8× vs vendor incremental growth (footprint/page-fault gap; gzip_chunk.rs:263-274
↔ IsalInflateWrapper); D2 unconditional per-iter process_ready_prefetches+harvest at consumer top
(chunk_fetcher.rs:1213,1218 ↔ none in ParallelGzipReader.hpp:575-643); D3 full prefetch-cache scan
on CLEAN branch (1715 ↔ vendor marker-only GzipChunkFetcher.hpp:513); D4 per-chunk throwaway
format! trace Strings with tracing OFF (1690,1697,1578); D5 chunk-0 full 32KiB zero window vs vendor
empty {} (560-561 ↔ GzipChunkFetcher.hpp:105); D6 ungated Instant::now() on hot path; D7
recycle_deferral 2-deep (CRC-race workaround, fragility signal). NEGATIVE/FAITHFUL (ruled out):
writev output, OverlapWriter OFF, CRC combine (no rescan), WindowMap.get zero-alloc, prefetch sizing
max(16,pool)/2×pool, ThreadPool T1 inline-deferred, window-publish handoff. Source-only by discipline
(box held by residual oracle); rankings reasoned not measured; Agent tool ABSENT ⇒ self-disproof only.
OWES Supervisor Opus gate + a measured perturbation of the cheap D2-D6 bundle. NOT advisor-vetted.


## VAR_VIII FULL-KERNEL INLINE-ASM ISOLATION BENCH — OWNER checkpoint [2026-06-09, worktree bench/var8-fullkernel, BENCH-ONLY]
Built the OWED full-kernel asm (back-edge `jmp 2b` INSIDE asm; F2 dist-gather + D AVX copy + F4 register-pin; Rust only at long-code/EOB/boundary) as `VAR_VIII_fullkernel` in `benches/engine_isolation.rs` (NO src/production touched). Frozen guest REDACTED_IP (bench-lock quiet, no_turbo=1, taskset -c 0, N=11 interleaved; released clean, NO orphans local+guest+neurotic). Built at origin tip 7bf26096 (d56cb0f5 local-only/unpushed; ancestor, every bench-relevant primitive byte-identical). **BYTE-EXACT: SHA_ALL_EQUAL=yes all 5 chunks vs scalar AND ISA-L; SELFTEST=PASS.** asm_frac 0.929-0.999 (median 0.938), reentries 3.4-4.3K vs VAR_VII 372-446K ⇒ DIS-1 spill confound REFUTED. RATE (med-of-med /ISA-L): VAR_VIII **0.667x** (188 MB/s; per-chunk 0.66-0.72), VAR_VI(LLVM) 0.582x, VAR_VII 0.29x. VERDICT = THIRD outcome: VAR_VIII MATERIALLY beats VAR_VI (+14.6%, sign-stable ⇒ KILL/plateau REFUTED, register-pinning IS a lever) BUT FAILS the 0.85 PASS bar (closes only ~20% of the VAR_VI→ISA-L gap). Production integration NOT authorized; engine does NOT reach ISA-L-class rate in pure-Rust+asm; BAR-1 not cleared. Detail in disproof-ledger.md (bottom). OWES Steward/Opus gate.

## OPEN-1 DISENTANGLEMENT (faithful-placement SIZED, ENGINE HELD) — OWNER checkpoint: the faithful runtime-placement slice at T4 is ~0 / TIE-to-WORSE (−34ms); the low-T lever is the GATED ISA-L engine swap, NOT placement. New byte-transparent oracle `--kind placement-marker` (seed REAL boundaries via inline p=1 capture + GZIPPY_SEED_NO_WINDOWS=1 → window-absent prefix HELD on pure-Rust marker engine; isal_chunks<=14 asserted, NO 14→17 leak). STEP-0 proof-of-binary: gzippy-isal HEAD, env-unset isal_chunks=14/14/16 T4/T8/T1, fallbacks=0/0/1, path=ParallelSM, bin b9eb0a73. Box frozen (bench-lock host-freeze, quiet-gate runnable<=2.0, no_turbo=1) + released clean (guest no_turbo=0, wd inactive); NO orphans (local+guest+neurotic pgrep clean). Self-disproof + pre-registered falsifier only (Agent tool absent) ⇒ OWES Steward/Opus gate. [2026-06-08, OWNER turn, worktree owner/disentangle-placement @ d56cb0f5]

**FROZEN-GUEST MEASUREMENTS** (REDACTED_IP, bench-lock --lock, no_turbo=1, governor=performance,
interleaved N=11, same-sink /dev/shm regular file, rg 0.16.0):

| config | T4 gzippy | rg | ratio | T1 gzippy | rg | ratio | invariant |
|---|---|---|---|---|---|---|---|
| production (same-sink, env-unset) | 546ms | 494ms | **0.906x** | 1022ms | 917ms | 0.898x | isal_chunks=14 sha=OK |
| placement-marker (boundaries seeded, ENGINE HELD) | 580ms | 490ms | **0.845x** | 1023ms | 918ms | 0.897x | isal_chunks=12-13 (≤14, NO leak), seed hits=0, sha=OK |

**FAITHFUL-PLACEMENT SLICE = production − oracle = 546 − 580 = −34ms (TIE-to-WORSE, within spread).**
FALSIFIER FIRES: placement (engine held) is NOT a faithful low-T lever at T4. The 231ms seed-all
gain was the GATED engine swap (14→17) + the uncounted free pre-pass — NOT runtime placement. The
oracle's 580ms independently reproduces the prior no-windows leg (581ms). T1 is IDENTITY (oracle ==
production; seed is a no-op when windows always present) = instrument self-test PASS. a-fortiori:
placement here was FREE (uncounted p=1 capture); the "counted at runtime" version is only slower, so
the verdict holds with margin. **CONCLUSION: the T4 low-T lever is the user-GATED ISA-L engine swap
on the window-absent prefix (LEV-4 clean-rate generalized), not a schedulable placement gap.** No
production code changed (only a measurement-harness `--kind`). OWES Steward/Opus sign-off.

## LOW-T RESIDUAL DECOMPOSITION (OPEN-1) — OWNER checkpoint: on the VERIFIED isal binary (STEP-0: isal_chunks=14/14/16 T4/T8/T1, fallbacks=0, env-unset, path=ParallelSM) the T4 residual SPLITS as DOMINANT marker-bootstrap COMPUTE(A)+alignment(C) [seed-all ceiling 549->318ms = 1.55x, sha-verified byte-perfect], NEGLIGIBLE FFI-handoff(B) [seed-all still routes 17 ISA-L chunks yet beats rg]. FOUND a harness bug: oracle.sh --kind clean-only is a NO-OP (SEED_WINDOWS=1 => open("1") ENOENT => silent production fallback) — prior "clean-only TIE" numbers are PRODUCTION-vs-PRODUCTION. NO production code changed; box frozen via driver --lock + released clean (no_turbo=0, LXCs thawed, wd inactive); NO orphans (local+guest+neurotic pgrep clean). Self-disproof only (Agent tool absent) => OWES supervisor Steward/Opus gate. [2026-06-08, OWNER turn, worktree owner/lowt-residual-decomp @ d56cb0f5, bin b9eb0a73]

**FROZEN-GUEST MEASUREMENTS** (REDACTED_IP, driver --lock, runnable_avg 1.00-1.75 <=2.0,
no_turbo=1, interleaved N=11, same-sink /dev/shm regular file, rg 0.16.0, T4 mask 0,2,4,6):

| config (T4) | gzippy | rg | ratio | coverage / sha |
|---|---|---|---|---|
| production (env-unset) | 549ms | 495ms | **0.902-0.904x** (x2) | isal_chunks=14 fb=0, sha=OK |
| seed-all clean-only (REAL capture+replay) | 318-325ms | 495ms | **1.486-1.555x** | seed hits=16 miss=0 flip=0 isal=17, **sha=028bd002…cb410f byte-perfect** |
| no-windows (boundaries only; A still paid) | 581ms | 497ms | 0.855x | marker-compute NOT elided |
| no-bounds (windows only; guess offsets) | 555ms | 495ms | 0.891x | ≈ production |

**THE SPLIT {marker-prefix(A) / FFI(B) / scheduling(C)}:**
- DOMINANT = (A) marker-bootstrap COMPUTE (pure-Rust u16 marker decode+resolve of the
  window-absent prefix), ENTANGLED with (C) alignment as the enabler — CO-PRIMARY. The whole
  231ms seed-all ceiling needs BOTH (no-windows AND no-bounds each give ~0 alone). Matches
  project_pregate_placement_is_dominant_lever.
- (B) FFI-handoff = NEGLIGIBLE: the 318ms seed-all run still hands ALL 17 chunks to ISA-L FFI
  yet beats rg — FFI cannot be a big term.
- (C) alignment alone = 0 (no-windows no help); matters only as A's enabler.
- **T1 (0.899x) is DIFFERENT**: seed is a NO-OP (sequential => windows always present => clean
  path already taken, inflate_wrapper=16, hits=0) => NO marker bootstrap at T1. T1's residual is
  the ENGINE symbol-rate + per-chunk FFI + serial output floor, bounded by REAL ISA-L (ocl_cf
  0.899x zero-margin). The low-T residual is THREAD-COUNT-STRUCTURED: T1 = engine/FFI/output floor
  (NOT bootstrap-closable); T4 = ADDS the large removable bootstrap term.

**FAITHFULNESS CAVEAT (load-bearing, rule 3):** seed-all is a masks-binder CEILING that
OVER-removes — it hands gzippy precomputed correct windows from an UNCOUNTED p=1 pre-pass, work rg
ALSO does at runtime (rg carries the SAME u16 marker machinery). So 1.55x is an UPPER BOUND, NOT a
reachable target. The FAITHFULLY-CLOSABLE mechanism = rg's window-map + in-place marker narrowing
(DecodedData.hpp) being faster/more-precise than gzippy's bootstrap — a faithful port, NOT a new
divergent path. This does NOT prove T4 reaches 0.99x by closing the bootstrap; it proves the
bootstrap is the DOMINANT low-T-at-T4 term and FFI is NOT, redirecting the lever from FFI/scheduling
to the marker-bootstrap engine+placement (additive to, independent of, the user-gated clean-tail asm).
Verdict: plans/lowt-residual-falsifier.md (self-disproof; OWES Opus + Steward sign-off).

## ISA-L DORMANCY RECONCILIATION — OWNER checkpoint: the contradiction is RESOLVED in favor of the BANKED picture. ISA-L is ACTIVE (14/14 fallbacks=0) in env-unset gzippy-isal PRODUCTION at HEAD; the "runtime-dormant / isal==native" fresh number MEASURED A MISLABELED NATIVE BINARY (DIS-13). Scorecard re-established first-hand. NO production code changed; box frozen+released clean, NO orphans [2026-06-08, OWNER turn, worktree isal-dormancy-reconciliation @ d56cb0f5]

**FROZEN-GUEST MEASUREMENTS** (REDACTED_IP, no_turbo=1, BENCH_LOCK=quiet runnable_avg 1.00-1.25,
watchdog armed→released clean; N=11 interleaved, same-sink /dev/shm regular file, sha==028bd002…cb410f,
path=ParallelSM asserted, env-scrubbed/oracle-UNSET, rg 0.16.0):

| cell | gzippy-ISAL (bin 2d317027) | gzippy-NATIVE (bin a42d4600) | rg |
|------|----------------------------|------------------------------|----|
| T1 | **1032ms = 0.899x** (1% spread) | 1525ms = 0.608x (1%) | 927ms |
| T4 | **547ms = 0.900x** (3%) | 652ms = 0.761x (3%) | 493-496ms |
| T8 | **361ms = 0.990x = TIE** (9%) | 410ms = 0.915x (6%) | 358-376ms |

**RUNTIME COVERAGE (GZIPPY_VERBOSE, env-unset, same binaries, sha=OK):** gzippy-ISAL
isal_chunks=16/14/14 (T1/T4/T8) fallbacks=1/0/0 — **ISA-L FIRES ON THE BULK OF 14 CHUNKS**.
gzippy-NATIVE isal_chunks=0 (structurally impossible to be non-zero — the native stub returns
Ok(false), gzip_chunk.rs:390-408). clean_flipped 1.9-2.0% on BOTH (it counts only marker-loop
pre-handoff clean bytes; the post-flip clean TAIL — the bulk — goes to ISA-L AFTER FlipToClean and
is NOT in that counter; "98% marker bootstrap" was a MISREAD).

**WHAT THIS OVERTURNS (the prior RESIDUAL-ATTRIBUTION checkpoint below is REFUTED on its load-bearing
facts):**
1. "ISA-L is RUNTIME-DORMANT (isal_chunks=0)" — WRONG. That was a gzippy-NATIVE binary mislabeled as
   gzippy-isal: the fresh "isal" counters (isal_chunks=0 flip_to_clean=12 finished_no_flip=4
   window_seeded=2) and wall (654ms=0.757x) are byte-for-byte the gzippy-NATIVE signature I reproduced
   (native T4 = 652ms=0.761x, identical counters). Mislabeled-binary class (rule 4 / DIS-13).
2. "isal T4 0.757x == native" — WRONG: isal T4 = 0.900x, native = 0.761x; isal is 14-48% faster at
   EVERY T. The banked ocl_cf 0.899x REPRODUCES (env-unset isal 0.900x).
3. "the asm target is the marker bootstrap for BOTH builds" — WRONG framing. The builds DIVERGE: the
   NATIVE clean tail is the pure-Rust inner loop (target = that loop, to capture what ISA-L proves
   capturable); the ISAL clean tail is already ISA-L (its residual is bounded by REAL ISA-L 0.899x,
   not closable by a better engine — its lever is the marker-prefix/FFI/scheduling residual, OPEN-1).
4. "placement/OPEN-2 within-spread (clean-only 10ms TIE)" — its baseline (654ms=0.757x) was the
   NATIVE wall; the oracle must be RE-RUN on the verified isal binary (OPEN-2 RE-OPENED).

**PIVOTAL ANSWER (supervisor):** the BANKED GOAL #2 picture is REAL — gzippy-isal ties rg at T8
(0.990x), loses 0.90x at T4 / 0.899x at T1, far closer than native (0.608/0.761/0.915). ISA-L is
NOT dormant. GOAL #2 is NOT overstated as to ISA-L activity (it IS overstated under BAR-1, which
requires >=0.99x at EVERY T — isal still loses T1/T4). The asm decision input: gzippy-NATIVE's
clean-tail inner loop is the engine target (LEV-1 ~0.14-0.16x at T4, now corroborated by the
env-unset isal-vs-native A/B); gzippy-ISAL is engine-bounded by real ISA-L and needs the
marker-prefix/FFI/scheduling residual instead. NO production code changed (the only edits are
plan docs + the disproof-ledger; the byte-transparent build.rs/comment fixes were already in the
tree). NO asm started. Verdict: plans/isal-dormancy-advisor-verdict.md (self-disproof; OWES a real
synchronous Opus pass + Steward sign-off — the Agent/subagent tool was NOT available this env).

## RESIDUAL-ATTRIBUTION + OPEN-2 — OWNER checkpoint: the "56ms non-engine residual" does NOT exist at HEAD; ISA-L is RUNTIME-DORMANT on silesia (isal_chunks=0); the T4 deficit is engine-dominated (marker bootstrap, 98% of body); placement/OPEN-2 is within-spread (clean-only TIE); build.rs lying comment FIXED [2026-06-08, OWNER turn, branch reimplement-isa-l @ d56cb0f5; box frozen+released, no orphans] [SUPERSEDED 2026-06-08 by ISA-L DORMANCY RECONCILIATION above — this checkpoint's core facts (isal_chunks=0, isal==native, marker-bootstrap-for-both) were a MISLABELED NATIVE BINARY; see DIS-13]

**FROZEN-BOX MEASUREMENTS** (guest REDACTED_IP, no_turbo=1, BENCH_LOCK=quiet, runnable_avg
1.25-1.50, watchdog armed then RELEASED clean; N=9 interleaved, same-sink regular file,
sha=028bd002...cb410f, path=ParallelSM; gzippy-isal binary bin_sha=ea36c8a3, HEAD d56cb0f5):
- gzippy-isal PRODUCTION (env unset = ISA-L build default) **T4 = 654ms vs rg 495ms = 0.757x**
  (spread 2%); **T8 = 406ms vs rg 373ms = 0.919x** (spread 3%).
- clean-only (GZIPPY_SEED_WINDOWS=1, ALL windows seeded) **T4 = 644ms = 0.767x** = +10ms TIE.
- engine-isolation (ocl_cf, GZIPPY_ISAL_ENGINE_ORACLE=1) **ABORTS coverage-zero**: isal_chunks=0.

**RUNTIME COVERAGE (the reframe):** flip_to_clean=12 finished_no_flip=4; window_present=2/18
(11.11%), window_absent=16 (88.89%); **isal_chunks=0 isal_fallbacks=2 — ISA-L FIRES ON ZERO
CHUNKS**; clean_flipped_bytes = **2.0% of body**; marker bootstrap owns **98%** at 85-87 MB/s.

**WHAT THIS OVERTURNS:**
1. The charter's CORRECTED PICTURE ("gzippy-isal clean tail already runs ISA-L, matches rg, low-T
   deficit is the residual-only at 0.885x") is REFUTED AT RUNTIME: ISA-L is dormant; gzippy-isal
   T4 = 0.757x == native (within spread). The SOURCE routes the clean tail to ISA-L (verified;
   build.rs comment fixed to say so), but at runtime there is almost no clean tail (2%) and ISA-L
   declines the few window-present chunks ⇒ runtime-vacuous on silesia.
2. The banked ocl_cf 0.899x / native 0.740x gate and its 0.159/0.101 SPLIT do NOT reproduce
   (isal_chunks 14→0). OPEN-5 FIRED. LEV-1/LEV-2 split STALE-AT-HEAD (ledger updated).
3. OPEN-2 (consumer-imminent eviction) is MOOT: clean-only removes the entire window-absent
   region and is a 10ms TIE ⇒ placement is slack; the eviction-discriminator instrument is NOT
   owed. (Even a window-present chunk is 98% marker work.)

**RESIDUAL ATTRIBUTION (T4, silesia, removal-oracle-backed):** deficit 0.757x→1.0 (~240ms) is
DOMINATED by the pure-Rust MARKER-BOOTSTRAP ENGINE (98% of body). Placement/window-absent/
FFI-handoff/clean-tail are each WITHIN-SPREAD (placement bounded <=10ms by clean-only). This is
LEV-4 (2.3x clean-rate) generalized to the marker body — the USER-GATED inner-loop asm question
for BOTH builds.

**PIVOTAL ANSWER (supervisor):** gzippy-ISAL CANNOT reach T4 0.99x by closing a residual alone.
There is no large non-engine residual at HEAD; ISA-L on the clean tail is runtime-vacuous (2%,
dormant). The asm target is the MARKER BOOTSTRAP (the 98%), not merely the clean tail. No
faithful non-asm convergence was landable (the dominant divergence is the gated inner loop; no
within-spread non-engine lever exists to faithfully port). NO production code changed; the only
edit is the byte-transparent build.rs comment fix (branch fix/buildrs-isal-comment) + ledger.
Host frozen for the window then RELEASED clean (no_turbo=0, frozen_now=[], wd inactive). Orphans:
verified clean local+guest+neurotic. Owes: a real synchronous Opus disproof pass + Steward
bankability sign-off (the Agent/subagent tool was NOT available this environment — self-disproof
recorded in plans/residual-attribution-advisor-verdict.md).


## CONVERGE-BOOTSTRAP-DIVERGENCE — LEADER checkpoint: the located window-absent divergence IS the USER-GATED inner-loop engine; the one NON-inner-loop candidate (OPEN-2 consumer-imminent eviction) is NOT-yet-actionable (decisive discriminator unrun); NO code changed [2026-06-08, LEADER turn, branch reimplement-isa-l @ d56cb0f5]
LOCATE (source-verified first-hand at HEAD d56cb0f5, no new measurement — bench guest perception was
NOT bench-locked this turn: procs_running=3, Plex+neighbors live, neurotic frozen_now=[] => any number
would be EXPLORATORY-only, not bankable):
- The dominant window-absent-bootstrap residual = the CLEAN u8 TAIL DECODER, the pure-Rust inner
  Huffman loop. NATIVE build (the 0.740x T4 1.0x-bar): FlipToClean -> `finish_decode_chunk_contig_native`
  (gzip_chunk.rs:1169-1193, :1223) -> `block.decode_clean_into_contig` (marker_inflate.rs:2071) =
  the ISA-L-LUT-ported multi-symbol fast loop. ISAL build: FlipToClean ->
  `finish_decode_chunk_with_inexact_offset` -> `StreamingInflateWrapper` = pure-Rust resumable.rs
  (DESPITE the name `isal_clean_tail`; build.rs:98-101 comment is STALE — the clean tail is pure-Rust,
  NOT ISA-L FFI; advisor-confirmed). The u16 marker PREFIX is already FASTER than rg (0.68x). This is
  the SAME divergence the window-absent-attribution turn proved causally (SLOW_MODE A/B +194/+248ms in
  the clean engine, freq-neutral, marker body unchanged) and the advisor UPHELD.
- => the located instruction divergence is the INNER-LOOP ENGINE INSTRUCTION RATE (pure-Rust symbol
  decode emitting more instructions than rg's ISA-L). Per charter USER-GATED BOUNDARY: converging it =
  the inner-loop asm rewrite = the USER'S GATED CALL. STOP-and-report.
- ASM-SUFFICIENCY READ (advisor-vetted asm-kernel-feasibility-advisor-verdict.md): engine alone is
  NOT sufficient for T4 0.99 — bounded by ocl_cf == ISA-L == 0.899x (LEV-1). A perfect igzip-class
  engine projects to a TIE only if PLACEMENT is ALSO closed (co-primary, LEV-5). So the 56ms residual
  is NOT purely engine-instruction; a full-engine asm port is NECESSARY-but-NOT-SUFFICIENT for T4.
- NON-INNER-LOOP candidate examined (charter job-b discriminate): OPEN-2 = consumer-imminent chunk
  eviction (prefetch-horizon-advisor-verdict.md), DISTINCT from DIS-6 (not offset-supply): a free
  worker exists at every stall yet the covering chunk is neither in-flight nor resident
  (NOT_RESIDENT=4, has_nearest_le_start=0, idle_cap 1.67-2.67 != 0). The DECISIVE discriminator
  (never-dispatched vs dispatched-then-EVICTED; parked-vs-unspawned idle; N>>3) is UNRUN => the
  faithful fix (retention-protect the imminent chunk vs deepen horizon) is NOT yet identified. This is
  a MEASUREMENT turn requiring a frozen box + new instrument, not a convergence I can land this turn;
  and it is co-primary, NOT alone-sufficient. NOT started (no half-baked EXPLORATORY build on a
  thawed box).
NO production code changed. No worktree needed (no edit). Orphans: NONE (local + perception + neurotic
pgrep clean; neurotic watchdog inactive, frozen_now=[]). Host left as-found (thawed, not locked).

## JOB 2 — ISA-L STORED/FIXED CLEAN-TAIL COVERAGE: gap MAPPED + advisor-vetted x2 => gap is REAL on small-block-dense input but the CITED port is DOUBLY WRONG; real fix = gzip_chunk until_exact accept (a NEW gated turn). NO production code changed; probes+fixtures committed in worktree [2026-06-08, OWNER turn, worktree isal-resync-stored-fixed @ d56cb0f5, commits 8c87cc24+7695463d]
SOURCE-VERIFIED first-hand: (1) the cited rapidgzip isal.hpp:392-405 is readBytes() (byte-aligned
FOOTER reader), NOT a block resync — the real rapidgzip mechanism is readStream() (isal.hpp:255-360,
a loop of isal_inflate to BFINAL). (2) The gzippy-VENDORED patched ISA-L (igzip_inflate.c:2386-2389
+ 2507-2510) ALREADY emits stopped_at=END_OF_BLOCK after decode_literal_block (stored) AND
decode_huffman_code_block_stateless (fixed). (3) gzippy decline site gzip_chunk.rs:302-333: the
until_exact path needs a boundary whose bit_offset == stop_hint EXACTLY; the inexact path needs one
at-or-past stop_hint else accepts only on end_bit<=stop_hint (BFINAL guard, 19add96c).

EMPIRICAL (guest x86_64 /tmp scratch — NOT the JOB-1 pinned bench root; gzippy-isal release binary
+ wrapper probes; src/tests/isal_stored_fixed_probe.rs):
- WRAPPER records boundaries on stored (6/6, 64/64) AND fixed (== block count), incl. the production
  copy-free _into path. NO nbounds=0.
- The ADVERSARIAL tiny-block fixture (40 MB make_btype01_heavy_data, flate2 L1 + SYNC_FLUSH every
  2 KiB => 18.2 MB gz, ~20,480 blocks): full production gzippy-isal byte-exact EVERY run, but
  isal_fallbacks = 1 / 11 / 46-48 / 101-104 at T2/T4/T8/T16 (STABLE x3; fallbacks EXCEED isal_chunks
  at T4+ => MAJORITY of tail chunks decline). The "common degrade" reproduces THERE.
- BENIGN on ordinary fixed-Huffman (flate2 L6: isal_fallbacks 0, one-off 1) and irrelevant to
  silesia (JOB 1: 0 fallbacks). ALL-STORED routes to a separate path=StoredParallel (never the
  clean-tail engine).
- ROOT-CAUSE probe on the EXACT tiny-block stream: decompress_deflate_from_bit_with_boundaries
  records **40,960 boundaries** (2048-byte cadence) => the production comment's "ZERO boundaries on
  stored/fixed" is EMPIRICALLY FALSE for SYNC_FLUSH input. The decline is the until_exact EXACT-match
  accept (stop_hint rarely == a boundary bit with dense blocks), NOT absent boundaries, NOT isal.hpp.

ADVISOR x2 (synchronous Opus disproof, plans/isal-resync-advisor-verdict.md): PASS-1 UPHELD-W-CAVEATS
my lean-to-reject but OWED an adversarial fixture (the until_exact tiny-block regime). PASS-2 (after I
built+measured it) UPHELD the DECISION but flagged "biggest risk = mislocated root cause" (production
comment says zero boundaries) — RESOLVED by the 40,960-boundary probe (my accept-logic map is
CORRECT; the comment is outdated for SYNC_FLUSH).

**SUPERVISOR — JOB 2: the coverage gap is REAL on small-block-dense (SYNC_FLUSH/flush-dense
streaming) fixed/stored input — NOT the phantom an early read suggested — but the user-cited fix
(isal.hpp:392-405 read_in resync) is DOUBLY WRONG: wrong function (footer reader) AND the FFI wrapper
already records boundaries (40,960 on the repro). The real fix = relax gzip_chunk's until_exact
EXACT-match accept to coalesce to the nearest clean EOB (faithful rapidgzip readStream), a
correctness-sensitive seed-path change (risks re-introducing the 19add96c over-decode mis-seed) that
warrants its OWN gated turn with byte-exact dual-sha + fuzz + the committed adversarial fixture as the
coverage gate. THIS turn DELIVERS: the source refutation, the gap-location map, and the adversarial
fixture that PROVES the gap (the repro the prior GOAL#2 advisor caveat asserted but never measured).
No production code changed (probes + fixtures only, worktree commits 8c87cc24 + 7695463d). Findings:
plans/isal-resync-findings.md. Host restored. NO orphan processes.**

## JOB 1 — LOW-T (T4) ENGINE GATE: the owed TIGHT quiet-box ocl_cf-T4 — MEASURED, advisor UPHELD-W-CAVEATS => PARTIAL (engine share >=0.159x, non-engine residual <=0.101x); NEITHER asm-rewrite started [2026-06-08, OWNER turn, branch reimplement-isa-l, HEAD d56cb0f5]
The owed <=5%-spread quiet-box ocl_cf-T4 vs native-T4 vs rg (the disambiguator the
prior turns could not get on a Plex-loaded box). Box bench-locked QUIET (Plex+7 noisy
LXCs frozen, INSTANTANEOUS procs_running gate 1.00-1.50 <=2.0). FIXED a measurement-only
harness bug first: _oracle_guest.sh's fallback-assert grepped `isal_oracle_chunks=` but
the binary emits `isal_chunks=N isal_fallbacks=M` (chunk_fetcher.rs:870-874) => the
engine-isolation kind hard-failed coverage-unreadable on EVERY prior attempt. Script-only
fix (binary untouched). Harness friction noted: the build-stamp inputs-fingerprint is only
stable immediately after `--build` (a bare rsync re-run yields a different fingerprint over
the 13172-file vendor set) => always pass `--build` (cache-hit ~0.11s).

MEASURED (frozen quiet guest REDACTED_IP, T4, interleaved best-of-N, same-sink ->/dev/shm
regular file, path=ParallelSM asserted, sha==028bd002...cb410f):
- ocl_cf-T4 (ISA-L engine oracle, bin b9eb0a73, isal_chunks=14 fallbacks=0): 545ms vs
  rg 490ms = **0.899x**, spread 3% (rg 4%) — TIGHT.
- native-T4 (pure-Rust production, bin 710a6dc): 652ms vs rg 482ms = **0.740x**, spread
  4% (rg 6%) — TIGHT.

VERDICT (PARTIAL, pre-registered tie-break #2; falsifier plans/low-t-gate-falsifier.md):
ocl_cf 0.899 < 0.99 => F-ENGINE-CLOSABLE NOT met (the engine ALONE does not reach parity
at T4 — even REAL ISA-L, the fastest engine, loses 0.899x). But ocl_cf-native = 0.159 >>
spread (~5x) => NOT F-NON-ENGINE (the two are decisively unequal). => PARTIAL: the engine
swap pure-Rust->ISA-L recovers a large share of native's deficit. ADVISOR-CORRECTED SPLIT
(synchronous Opus disproof, plans/low-t-gate-advisor-verdict.md, UPHELD-W-CAVEATS; the
ocl_cf "ISA-L ceiling" is a BLEND, source-confirmed gzip_chunk.rs:128-131,196-223 — only
the clean 32KiB-window continuation goes through ISA-L FFI, the markered prefix + chunk-0
bootstrap STAY pure-Rust): **engine share >= 0.159x (UNDERESTIMATE)**, **non-engine residual
<= 0.101x / ~55ms (UPPER BOUND — contains marker-prefix pure-Rust engine + per-chunk ISA-L
FFI/handoff, NOT pure scheduling)**. ZERO-MARGIN RISK: an asm engine's best case == ISA-L
== 0.899x at T4 (never 0.99 alone) => parity needs BOTH levers fully realized to land ~1.0.

NAMED NEXT LEVERS (both gated, NEITHER started — asm is user's call):
(1) full-kernel hand-asm engine rewrite to CAPTURE in pure-Rust the >=0.159x share ISA-L
    proves capturable (justified by the FFI-off-decode-graph goal, NOT because only asm can
    close it). Bounded by ocl_cf 0.899x — cannot reach 1.0 alone.
(2) the <=0.101x non-engine residual = scheduling/bootstrap/FFI-handoff; LOWER-RISK, helps
    BOTH builds. Candidate project_confirmed_offset_prefetch_gap CANNOT be sized from 0.101
    until the marker-prefix-engine + FFI buckets are separated (advisor-owed: an oracle that
    also ISA-Ls the marker prefix, or an FFI-overhead null run).

**SUPERVISOR — JOB 1 gate CLOSED: the low-T deficit is PARTIALLY engine (>=0.159x) +
PARTIALLY non-engine (<=0.101x, upper bound). The full-kernel asm rewrite remains GATED on
your call; it can only CAPTURE the engine share, not reach 1.0 alone (zero-margin). NO
production change landed (measurement-only script fix). Verdict: plans/low-t-gate-advisor-verdict.md;
falsifier: plans/low-t-gate-falsifier.md. Host auto-restored thawed. NO orphan processes.**

## JOB-2 SYNC_FLUSH RESYNC PORT — OWNER checkpoint: the gap's ROOT CAUSE was a RESERVE UNDER-SIZING bug (writable_tail_reserve), NOT the coalesce accept (the inexact path was already a faithful readStream port; the exact-bit decline is faithful). FIXED byte-transparent; dual-sha both features byte-IDENTICAL; differential gate 10/10; coverage declines reduced (44→37 @T8) + a worker PANIC removed. Built+ran the REAL ISA-L path LOCALLY via Rosetta (NO neurotic — Plex window honored). [2026-06-08, OWNER turn, worktree isal-resync, branch isal-resync-stored-fixed @ 7695463d+]

LOCAL ENABLER: `git submodule update --init vendor/isa-l vendor/isal-rs` + `cargo {build,test}
--target x86_64-apple-darwin --features gzippy-isal` (nasm, RUSTFLAGS=-C target-cpu=x86-64-v2)
BUILDS+RUNS the real ISA-L FFI clean-tail under Rosetta — full byte-exact + coverage iteration
without the box (extends reference_local_x86_tests_rosetta to the isal feature). NO neurotic
ssh/rsync/freeze this turn.

TWO-COLUMN MAP (source-verified first-hand, vendor isal.hpp:253-385 readStream +
GzipChunk.hpp:282-385): rapidgzip INEXACT coalesces FORWARD to the first clean block start at/past
untilOffset (skips final/FIXED). gzippy's PURE-Rust StreamingInflateWrapper (gzip_chunk.rs:800-810)
is ALREADY a byte-for-byte port of that. The ISA-L-oracle accept (gzip_chunk.rs:290-333) mirrors it:
inexact picks first recorded boundary >= stop_hint and NEVER over-decodes (declines if none + end_bit
> stop_hint — charter rule honored); exact requires a boundary == stop_hint else declines to bit-exact
pure-Rust, mirroring vendor decodeChunkWithInflateWrapper's land-exactly-or-fail contract. So there
was NO non-faithful accept to "port".

ROOT CAUSE: `SegmentedU8::writable_tail_reserve` did `reserve(min_spare - (cap-len))`; Vec::reserve
measures from len, so that UNDER-requests and no-ops when min_spare>cap but the delta<existing spare
(len=49 KiB, cap=16.6 MiB, min_spare=16.5 MiB → reserve(704 KiB) no-ops → SHORT). The copy-free ISA-L
FFI then got a too-small slice on dense tiny-block SYNC_FLUSH input. FIX = `reserve(min_spare)`.
- DEBUG/TEST: debug_assert FIRED → worker PANIC → pipeline HANG (headline correctness win: removed).
- RELEASE: assert out; under-reserve returns None (decline, SAFE byte-exact) ⇒ SPURIOUS DECLINES.
- Only production caller is the ISA-L oracle (gzip_chunk.rs:274); the other (:1017) is the OFF
  fold_nodrain knob ⇒ gzippy-NATIVE byte-UNAFFECTED (isal_chunks=0; native_fold_parity 4/4 green).

PROOF (LOCAL Rosetta x86 release binaries, tinyblocks 10.7-10.9 MiB > parallel gate, path=ParallelSM):
- DUAL-SHA both features × T1/T4/T8 ALL == raw `3af314a4…51d18` (native ≡ isal, byte-identical).
- COVERAGE A/B (flate2 fixture, both byte-exact): T8 isal_chunks=22 fallbacks 44(OLD)→37(NEW);
  T4 10/17→10/15. Declines reduced; surviving ones are the FAITHFUL exact-bit case. In DEBUG/TEST
  the delta is HANG/PANIC → byte-exact isal_chunks=37.
- isal_tail_parity differential gate 10/10 MATCH 0 diverged (both until_exact values); routing 28/28;
  native_fold_parity 4/4; isal_stored_fixed_probe 4/4 (new isal_coverage_on_tiny_blocks gate green).
  7 full-suite fails are PRE-EXISTING Rosetta/env flakes (reproduced on a stash-baseline).

DELIVERED (production): segmented_buffer.rs reserve fix (byte-transparent native, correctness+coverage
isal). TEST: gzip_chunk.rs synthetic-corpus fallback so the silesia differential runs locally too;
isal_stored_fixed_probe.rs coverage gate + portable materializer. NOT advisor-vetted (Agent tool
absent in owner env) — OWES supervisor's Opus advisor + correctness gate. NO neurotic touched, NO
wall runs. Orphans: my hung test binaries (parallel-test-hang deadlock) KILLED; pgrep clean for my
procs. Findings: plans/isal-resync-findings.md (JOB-2 PORT TURN section).


## CACHE-RESIDENCY LITERAL ARCHITECTURE — MET (the two SCOPED items closed, byte-exact, re-measured) [2026-06-08, OWNER turn, branch reimplement-isa-l, HEAD 8d4f20f7+]
Completes the cache-residency clause's literal architecture from the prior turn's
SCOPED list (plans/cache-residency-verdict.md §SCOPED). Full verdict:
plans/cache-residency-architecture-met.md.
- (1) DIST CACHE SHRUNK 128 KiB→8 KiB/thread (commit 8d4f20f7, byte-exact). Ported
  vendor's bounded-LUT `HuffmanCodingShortBitsCached` (huffman/…ShortBitsCached.hpp)
  for the production distance path: LUT_BITS_COUNT=12, Symbol=u8 ⇒ (u8,u8)*2^12 =
  8 KiB (was [(u8,u16);1<<15] = 128 KiB, itself 2× the vendor 64 KiB Symbol=uint8_t
  cache). decode_long fallback for >12-bit codes. New file
  src/decompress/parallel/huffman_short_bits_cached.rs. PER-THREAD 278.8→158.8 KiB
  (target ~150 → MET).
- (2) SegmentedU8 DecodedData: BOTH levers of the original port (2b8bfae/442e93e2)
  ALREADY in tree (data:SegmentedU8 + in-place narrowed_len). The only un-landed
  piece — physical segment-list backing — was DELIBERATELY/FAITHFULLY superseded by
  e8c03110 (contiguous DecodedData, DecodedData.hpp:278-289); re-landing it would
  REGRESS the ContigFold copy-free tooth (decode_clean_into_contig's base+cap
  write) the verdict forbids regressing AND diverge from vendor. REJECTED with
  mechanism (charter rule #7b). Item closed: landed where compatible, rejected where
  it conflicts.
- (3) output_ring 128 KiB KEPT (faithful vendor m_window16 floor).
RE-MEASURED (frozen guest REDACTED_IP, gzippy bin_sha e32829a8, sha==028bd002…):
  RSS-vs-T +204.5% (was +216%); per-thread 158.8 KiB; MPKI LLC-miss 0.204 (was
  0.205, < rapidgzip 0.367), L1d-miss 4.03 (was 4.55, < rg 5.89) — NO regress,
  improved; ballast control linear 16.0/16.0/15.8 thr/MiB; WALL parity.sh
  T8=0.899× TIE / T16=0.921× TIE (sha=OK) — NO wall regress. Dual-sha both builds:
  gzippy-native + gzippy-isal (x86 via Rosetta) byte-exact @L1/L6/L9, path=ParallelSM.
ADVISOR: claude -p (Opus) SPAWNED OK — synchronous disproof on dist-cache
byte-exactness (5 attack vectors) → claim SURVIVES, no falsification (new≡base≡old;
only error-timing on truncated streams differs, outside legal-stream scope).
HARNESS CAVEAT: rss_vs_t.sh MPKI label prints `-nan` (awk divisor formatting bug
with cpu_core/-prefixed counter under paranoid=4); RAW counters valid, MPKI
computed from them. Follow-up: fix the awk label.

## CACHE-RESIDENCY MANDATE — MEASURED (the never-measured DECISIVE clause closed) [2026-06-08, OWNER turn, branch reimplement-isa-l]
The instrument-dead admission at :1804-1815 is RESOLVED. mem_stats RE-HOOKED to the
real native per-thread working set (`marker_inflate::Block`/`BOOTSTRAP_BLOCK`, via
`on_block_active(block.heap_bytes())` in `marker_decode_step_vendor_block`) — the
old hook (staged_bits `take_staging_box`) was DEAD on native after the fold and read
threads=0. New `Block::heap_bytes()` + `LutLitLenCode::heap_bytes()` +
`HuffmanCodingReversedBitsCached::heap_bytes()` (counters only, byte-transparent).
Built `scripts/bench/rss_vs_t.sh` + `_rss_vs_t_guest.sh` (the missing deliverable
at :1662): host-freeze bracket, sha-verify EVERY run, path=ParallelSM assert,
RSS-vs-T + per-thread byte accounting + ballast positive-control + perf MPKI
(validated-instrument-first).
MEASURED (frozen guest REDACTED_IP, 16-core RaptorLake i7-13700T, gov=performance
no_turbo=1 runnable_avg=1.75, gzippy bin_sha 8a3524088d47fcd9, feature=gzippy-native,
every decode sha == 028bd002…cb410f):
- INSTRUMENT VALID: threads=16 (was 0); OFF==identity; 881 lib tests pass; ballast
  positive control recovers per-thread slope LINEARLY (16.16/15.80/16.01
  threads-recovered per +1 MiB at 0→8→16→32) == the 16 worker threads; perf ballast
  control moved cache-misses 1.23× / LLC 1.09× monotonically.
- PER-THREAD working set = 278.8 KiB: output_ring 128 KiB + dist_hc.code_cache
  128 KiB + lut_litlen 22.5 KiB + misc 0.3 KiB. SHARED read-only tables only 18 KB.
- RSS-vs-T (min-of-5, KiB): gzippy 122800/322160/388072 @T1/8/16 vs rapidgzip
  67164/211444/312136 = 1.83×/1.52×/1.24×. gzippy RSS +216% T1→T16 (NOT flat).
  (The wall-progress 3× @T8 reading is PRE-fold; post-fold the gap is 1.24–1.52×.)
- MPKI @T16: gzippy LLC-miss 0.205 / L1d-miss 4.547 / cache-miss 1.942 — BELOW
  rapidgzip's 0.379 / 5.766 / 2.520, though gzippy runs 1.48× more instructions.
VERDICT: mandate NOT fully met on the working-set-size / RSS-flatness clauses
(278 KiB/thread, +216% RSS-vs-T), but its PURPOSE — hot-in-cache / low MPKI — is
ALREADY satisfied (gzippy MPKI < rapidgzip; the 278 KiB streams with locality, no
measured miss penalty). Shared-tables clause PARTIAL (fixed-Huffman shared;
per-block dynamic tables + the 128 KiB dist cache are per-thread). Cross-checks the
banked 5-refutation x86 STOP: footprint is wall-SLACK, so the architecture work is
mandate-justified (design goal), wall-no-regress-guarded.
SCOPED for NEXT turn (do NOT start): (1) shrink/share dist_hc.code_cache (CACHE-LEN
2^15=32K entries vs ~30 dist symbols → 2-level / smaller cache, 278→~150 KiB); (2)
re-land the SegmentedU8 DecodedData (commit 2b8bfae, −29% RSS @T8, wall-TIE) for
RSS-flatness; (3) output_ring is a faithful vendor floor (m_window16) — do NOT
shrink. Each byte-exact + re-measured on rss_vs_t.sh. Full verdict:
plans/cache-residency-verdict.md.

## GOAL #2 SHIPPED — gzippy-isal NOW ROUTES THE PRODUCTION CLEAN TAIL THROUGH REAL ISA-L FFI (faithful rapidgzip WITH_ISAL), no longer env-gated. Byte-exact both features; full suite + net + differential gate green; advisor UPHELD-W-CAVEATS. TIES rg at T8 (1.030×), far closer than native at low-T [2026-06-08, OWNER turn, branch reimplement-isa-l, HEAD 19add96c]

The /goal mandates TWO shipped implementations. gzippy-native (pure-Rust, no-FFI) was done; gzippy-isal existed ONLY as the GZIPPY_ISAL_ENGINE_ORACLE measurement knob. THIS TURN promoted it to a PRODUCTION path.

CHANGE (commit 19add96c, 2 files: gzip_chunk.rs + chunk_fetcher.rs): on the `isal_clean_tail` (gzippy-isal) build, `finish_decode_chunk_impl` routes the clean tail (post-32KiB FlipToClean bulk + window-seeded chunks) through real ISA-L FFI `decompress_deflate_from_bit_into` — mirroring rapidgzip `finishDecodeChunkWithInexactOffset<IsalInflateWrapper>` (GzipChunk.hpp:440-444 + :520-526); the <=32KiB markered prefix stays the pure-Rust marker engine (deflate::Block). `isal_engine_oracle_enabled()` now defaults to `cfg!(isal_clean_tail)` (env 1/0 still forces on/off); `finish_decode_chunk_impl` gained `allow_isal` (the differential gate's pure LEFT passes false). gzippy-native UNCHANGED (isal_chunks=0, pure FOLD, byte-exact).

TWO BYTE-EXACTNESS FIXES the silesia-only oracle never exposed (caught by the btype01-heavy + multi-subchunk routing traps): (1) OVER-DECODE — ISA-L's END_OF_BLOCK stop doesn't fire on stored/fixed blocks ⇒ nbounds=0 ⇒ inexact coalesce committed a 2MB over-decode past stop_hint (mis-seeded next chunk). Fix: accept `(end_bit,written)` only when `end_bit<=stop_hint` (genuine BFINAL); else DECLINE to byte-exact pure-Rust. (2) SUBCHUNK SPLIT — credited keep_len up front, hiding per-segment growth so `append_block_boundary_at` never split the ISA-L tail (UNSPLIT trap). Fix: incremental `note_inner_decoded_bytes` per [prev,boundary) segment (vendor GzipChunk.hpp:321). Byte-transparent.

MEASURED (x86_64 ISA-L guest, parity.sh bench-lock, interleaved N=11, sha vs rg 0.16.0): T1 1019ms=0.904×(tight) | T4 554ms=0.885× | T8 363ms=1.030×=**TIE**. ISA-L coverage 14/14 fallbacks=0 @ T4/T8; 16/1 @ T1 (the 1 = until_exact exact-stop, byte-exact net). Reproduces TIME-ACCOUNTING ocl_REAL exactly (T1 1015/T4 548/T8 365). vs gzippy-native (0.608/0.755/0.885×): the ISA-L reference is far closer at low-T and ties at T8.

CORRECTNESS: silesia dual-sha 028bd002…cb410f @ T1/T4/T8 BOTH features; full lib suite green under GZIPPY_POISON_RESERVE=1 (the lone non-green = un-synced test_data/alice.txt fixture, env not logic; fd_vectored_write reader-death = pre-existing load-flake); hardened isal_tail_parity differential gate 10/10 (until_exact false+true: bytes/len/final_bit/crc/boundaries); btype01 + coordinator + unsplit + single-member routing green.

ADVISOR (synchronous disproof, plans/isal-production-advisor-verdict.md): UPHELD-WITH-CAVEATS. All byte-exactness/fallback break attempts FAILED (declines return BEFORE any commit = clean handoff; over-decode accept provably only BFINAL within ~64 bits of slice_end; incremental note telescopes to exactly keep_len). CAVEAT (honest scoping, non-corrupting): "faithful WITH_ISAL" holds where ISA-L's EOB-stop contract holds — on a stored/fixed-block-heavy corpus declines become common and gzippy-isal degrades to gzippy-native (byte-exact, zero ISA-L coverage; the TIE-vs-rg claim holds on all-dynamic data). Fully closing needs porting rapidgzip's read_in resync (isal.hpp:392-405) for non-dynamic blocks — OUT of scope, correctness unaffected.

**SUPERVISOR — GOAL #2 (second shipped implementation) COMPLETE. gzippy-isal is the at-parity reference: ties rg at T8, 0.88-0.90× at low-T (vs native 0.61-0.76×). Byte-exact, faithful WITH_ISAL on the parity corpus, advisor-vetted. NEXT (owed, optional): port the ISA-L non-dynamic read_in resync to extend faithful ISA-L coverage to stored/fixed inputs (closes the advisor caveat). The gzippy-native 1.0× engine A-vs-B fork (below) is unchanged + independent.** GUEST: /root/gzippy-bench (pin), gzippy-isal bin 44c81326f81c8da9. Host restored (no_turbo=0, watchdog disarmed by parity.sh). NO orphan processes. Verdict: plans/isal-production-advisor-verdict.md.

## SUPERSEDED — TOOLING-AUDIT FIXES LANDED + A-vs-B DECISION INPUTS RE-MEASURED (hardened oracle path, interleaved RATIO, advisor UPHELD-W-CAVEATS) → ocl_cf was PESSIMISTIC, now ≈TIE/≥0.945×; low-T LOSS is real+engine-bound but "engine-CLOSABLE" not yet measurement-clean; absolute split BLOCKED-on-load [2026-06-08, OWNER turn, branch reimplement-isa-l, HEAD f216c691]

Owned the audit's job: harden the under-guarded oracle.sh path to the parity.sh bar, fix the 64MiB-reserve confound, clean the guest disk, then re-measure the A-vs-B decision inputs on a quiet box.

HARDENED (commit e4389f05 + f216c691, scripts measurement-only): _oracle_guest.sh now matches the parity spine — env-scrub ALLOWLIST, content-fingerprint stale-binary guard (oracle.sh --build stamps it), host-freeze HARD-FAIL on a readable thaw (was WARN-only = the source of ocl_cf's 0.945↔0.989 drift), loadavg readback HARD-FAIL >2.0 (ALLOW_LOAD=1 → ratio-only), IN-SCRIPT isal_oracle_fallbacks==0 assert from the GZIPPY_VERBOSE sidecar (was hand-checked). The host-freeze guard CAUGHT the box THAWED (no_turbo=0, watchdog expired) — re-froze.

RESERVE-CONFOUND FIX (audit A5 / number #2): the ISA-L oracle reserved a flat 64MiB/chunk where production grows incrementally to ~chunk size; now (compressed_span × 8).max(4MiB).min(64MiB) ⇒ ~38MiB for a 4.7MiB T8 chunk. Under-reserve is SAFE (FFI None → pure-Rust fallback → counter → assert VOIDs loud — proven at T1 where the assert VOIDed on the window-absent bootstrap chunk). Byte-exact (gated GZIPPY_ISAL_ENGINE_ORACLE, OFF==identity).

GUEST DISK: 90% (3.4G free) → 81% (6.4G free). Removed /root/gzippy/{target,benchmark_data} (non-pin, rebuildable), stale /tmp build trees + ~6G decoded-output dumps. Pin /root/gzippy-bench + corpus untouched.

RE-MEASURE — BLOCKED-on-load for ABSOLUTES; RATIO captured (interleaved jitter-immune, sha-OK, fallbacks==0 in-script): the host (neurotic) was Plex-loaded the whole turn (loadavg bouncing 1.6↔11 as transcodes arrived; polled bounded ~25min, never reliably quiet; Plex runs on the host, outside clock_freeze's reach). RATIO results:
- ocl_cf T8 = **≈0.99× / ≥0.945× rg** (14/0 clean; one run 0.997×@load1.63; reproduced 0.98-1.00× ×3) — UP from banked 0.945× PESSIMISTIC, confirming the audit prediction.
- gzippy-NATIVE (production): T8 0.885× (TIE, 19% spread), **T4 0.755× LOSS (3% spread, tight)**, **T1 0.608× LOSS (4% spread, ==inner-loop 0.55-0.60× ⇒ ENGINE-BOUND)**.
- ocl_cf low-T: T4 ~0.900× (16% spread, LOOSE), T1 VOID (bootstrap chunk window-absent ⇒ no ISA-L ceiling).
- 36/21 split: NOT re-measured (needs a quiet box; still OWED). Do NOT bank 36/21.

ADVISOR (synchronous disproof, plans/oracle-hardening-advisor-verdict.md): UPHELD-WITH-CAVEATS — hardening+reserve fix byte-exact+safe; bank ocl_cf "≈TIE/≥0.945×" not "0.997×"; **the low-T LOSS is real+engine-bound (strong), but "a better engine CLOSES it" is NOT established — ocl_cf-T4's 16% spread admits "engine swap does nothing, residual non-engine"; T1 ocl_cf VOID ⇒ T1 can't prove closability.** OWED to make option-B clean: ONE tight (≤5% spread, quiet box) ocl_cf-T4.

**SUPERVISOR GATE — A-vs-B inputs VALIDATED in direction with ONE owed tightening. ocl_cf is NOT 0.945× (≈TIES rg) ⇒ option B's T8 upside is ~nil. The case FOR B rests ENTIRELY on the LOW-T cells: native LOSES at T1 (0.608×, engine-bound) + T4 (0.755×) where the engine is NOT slack-masked. BUT the advisor REFUTED "engine-closable" as not-yet-clean at T4. NEXT OWED before (B) is authorizable: ONE tight (≤5%, quiet box) ocl_cf-T4 vs native-T4 vs rg — if ocl_cf-T4 ties rg with native losing ⇒ B justified at low-T; if ocl_cf-T4 ≈ native ⇒ low-T gap is NON-engine, close the engine chapter + attack low-T scheduling. The owner could not get a quiet box (active Plex). Do NOT start the full-kernel asm — user's call.** GUEST: /root/gzippy-bench (pin), gzippy-isal bin b3d9435587186f27 (factor-8) / gzippy-native bin 9790e8f3. Hardened drivers scripts/bench/{oracle.sh,_oracle_guest.sh}. Host re-frozen (no_turbo=1, watchdog ttl 3600s). NO orphan processes. Verdict: plans/oracle-hardening-advisor-verdict.md.

## SUPERSEDED — PHASE 1 (source-map) + PHASE 2 (isolation prototype + measure) of the INLINE-ASM igzip fork → DECISIVE NO-GO; user-fork escalated (advisor disproof ×2, pass-2 signed off) [2026-06-08, OWNER turn, branch reimplement-isa-l, HEAD 690941f3]

Executed the charter's TIERED Phase-1+Phase-2 (prove-before-the-big-build).

PHASE 1 SOURCE-MAP (read-only subagent, cited igsip_decode_block_stateless.asm:507-627): what
the hand-asm does that LLVM does NOT emit — F1 one-iteration-ahead literal-table gather across the
back-edge (asm:540), F2 speculative dist gather before the lit/len branch (asm:550-552), F3
unconditional flag-free SHLX/SHRX refill+consume (IN_BUFFER_SLOP=8, asm:528-547), F4 loop state
pinned in callee-saved GPRs (asm:108-136), C 8-byte speculative packed-literal store (asm:518), D
16-byte MOVDQU + distance-doubling copy (asm:603-627), E one slop-gate per iter (asm:488-512).
VAR_V/VAR_VI already do C/D/E + a Rust PRELOAD; VAR_VII targets F1/F3/F4/C in `core::arch::asm!`.

PHASE 2 PROTOTYPE (benches/engine_isolation.rs VAR_VII, committed 0c769c14→690941f3): inline-asm
literal-run hot loop, regs pinned, unconditional SHLX refill (mirrors gzippy Bits bitsleft=len|56),
12-bit gather, 8-byte packed store, one slop-gate; exits to Rust per back-ref/long-code then
RE-ENTERS the asm. BYTE-EXACT SHA_ALL_EQUAL=yes on all 5 swept clean silesia chunks vs scalar AND
ISA-L. GZIPPY_VII_COVERAGE: asm emits 57-99% of bytes (median ~0.65-0.74).

PHASE 2 MEASURE (GUEST native x86_64, frozen host, taskset core 0, interleaved best-of-11, vs ISA-L
oracle): ISA-L 283 MB/s (1.000×) | VAR_VI LLVM 168 (0.594× = LLVM ceiling) | VAR_VII asm 78
(0.276×, ~0.75× of NAIVE SCALAR). TWO brackets: leading-run-only 0.55×, re-enter-per-symbol 0.28× —
rate FALLS as asm coverage rises ⇒ the per-symbol asm↔Rust re-entry (LLVM barrier ×300-460K/chunk,
4 regs spilled to `bits`) dominates. FALSIFIER ⇒ PLATEAU/NO-GO.

ADVISOR ×2 (independent disproof, synchronous): pass-1 broke the first cut (asm ran once/block over
the leading run, 0.89% coverage = under-powered) → OWED a coverage counter + dominant-path asm; owner
delivered both. pass-2 tried 3 breaks, all FAILED → NO-GO EARNED, escalation correctly triggered;
REVERSED its own pass-1 "full-kernel VAR_VIII before escalating" — the full-kernel upside is already
bounded by the ocl_cf 0.945× removal oracle, so VAR_VIII can only CAPTURE not re-decide the fork.
SHARPENING: T8 engine slack-masked → accept; but goal #1 = no-FFI parity across thread-count incl.
LOW-T where the engine is NOT masked (0.55-0.60× ISA-L) — full-kernel asm justified IFF a low-T
engine-bound CLOSABLE cell loses to rapidgzip.

**SUPERVISOR GATE — INLINE-ASM FORK RESOLVED (NO-GO on the transliteration approach, advisor-signed).
The only remaining engine lever is a full-kernel asm (re-write ISA-L by hand) bounded by ocl_cf 0.945×.
ESCALATE: (A) ACCEPT ~0.86-0.945× pure-Rust and close the engine chapter, OR (B) authorize a
multi-session full-kernel asm — justified ONLY if a LOW-T engine-bound CLOSABLE cell loses to rg
(T8 is slack-masked, does NOT justify it). NEXT OWED before (B): a T1/T4 parity.sh whole-system cell
vs rg to test for a closable-and-losing engine-bound cell. NO production change (bench VAR_VII + plan
docs; byte-exact preserved). Briefs/verdicts: plans/phase2-inline-asm-brief.md +
plans/phase2-inline-asm-advisor-verdict{,-pass2}.md. Host auto-restored frozen. NO orphan processes.**

## SUPERSEDED — C2 NON-ENGINE RESIDUAL LOCALIZED + BOUNDED on ocl_cf → ALL C2 SUB-TERMS FLAT-OR-SMALL; the 21ms is NOT a faithful low-risk scheduling tooth (3 removal oracles, advisor signed off ×2) [2026-06-08, OWNER turn, branch reimplement-isa-l, HEAD f98af1f]

Owned the supervisor's job: localize the 21ms non-engine residual on the ocl_cf (ISA-L engine) path,
bound it with a removal oracle, faithful-fix if found. Froze the host (was thawed on arrival), built
gzippy-isal @ f98af1f, re-confirmed ocl_cf 0.966-0.989× rg (residual ~5-14ms; banked 21ms reproduces
structurally, smaller absolute on this loaded host).

LOCALIZED (fulcrum_total + consumer_block_decompose.py, consumer tid=1, native + ocl_cf): swapping the
clean engine to ISA-L barely moves consumer_wall (~5ms) and does NOT shrink DECODE_WAIT (~64% both).
consumer.writev (~0.12-0.13s) dominates SERIAL; resolve/publish spans ~0ms (apply_window already off the
consumer crit path = faithful).

BOUNDED (3 removal oracles on ocl_cf, pre-registered falsifiers, interleaved, byte-exact A):
1. PERFECT_OVERLAP (dispatch-DEPTH): A→B −0.2% FLAT (warm_hit_frac 0.882) ⇒ head-of-line DEPTH DEAD.
2. warm_miss cause: guard-rejects=1 of 2 misses ⇒ interior-reuse/overshoot fires on 1 chunk, negligible.
3. SEEDFULL bootstrap-removal (masks-binder CEILING): A−B ≤8ms; ocl_cf+seedfull STILL 0.983-0.985× not
   1.0× ⇒ ~6ms output-floor residual SURVIVES full bootstrap removal (fresh removal confirmation).

VERDICT (advisor disproof ×2, plans/c2-residual-disproof-verdict.md — SIGNED OFF, "argued→removed",
EARNED leg-by-leg): scheduling NULL, interior-reuse 1-chunk, bootstrap ≤8ms engine-class, dominant
residual = the (prior-established) shared writev/bandwidth OUTPUT FLOOR. project_confirmed_offset_prefetch_gap
is REFUTED as the 21ms (DEPTH dead by removal; TARGETING 1-chunk).

**SUPERVISOR GATE — C2 located + bounded NON-CLOSEABLE-FAITHFULLY (prompt case-4: validated removal +
no faithful wall-moving mechanism). NO production fix landed (Rule-3 removals + plan docs only; no src/
change, byte-exact preserved). FORK INPUT UPDATES: C2 removed as the cheaper faithful alternative — the
"KEEP-GRINDING C2" premise is substantially weakened; remaining levers are (a) the engine inline-asm fork
(~36ms) + (b) a small shared/irreducible output floor. Does NOT by itself escalate the engine fork
(residual small + partly shared, not provably engine-only at loadavg ~4). NEXT (owed): quiet-box re-measure
to tighten the bootstrap/floor split + turbo-freq confirmation of the 36/21 split → then the fork is a clean
"inline-asm-or-accept-0.945-0.98×" decision.** Briefs: plans/c2-residual-localization-brief.md; verdict:
plans/c2-residual-disproof-verdict.md. Drivers: scripts/bench/{_oclcf_overlap_bound,_c2_advisor_owed}.sh.
GUEST /root/gzippy-bench, gzippy-isal bin sha 2f86676. Host left frozen (watchdog auto-restores +5400s).
NO orphan processes.

## SUPERSEDED — CLEAN ENGINE-RATE CEILING MEASURED + VALIDATED (matched comparator) → engine plateau REAL & bounded; FORK NOT escalated (advisor: KEEP-GRINDING C2, not inline-asm) [2026-06-08, OWNER turn, branch reimplement-isa-l, HEAD a884fa7b+wt]

MERGE green (886 lib pass, poison-on; 3 prior failures fixed). Fixed the merged parity.sh/oracle.sh
which could NOT build (4 script fixes: rsync --delete-excluded→--delete for macOS openrsync;
RSYNC_PATHS += crates examples; un-scrub GZIPPY_BIN in the contamination allowlist; fingerprint +=
crates examples). Froze host via neurotic /root/clock_freeze.sh. Killed a stale 970MB orphan test
binary before measuring.

DECISIVE (frozen host, MATCHED comparator both→/dev/shm regular-file sink, interleaved best-of-N min,
sha-verified, window-absent-preserving):
- rapidgzip 0.16.0          376-383 ms  1.000×
- ocl_cf (ISA-L clean eng)  404 ms      0.945×  ← validated CEILING (coverage 14/0, symmetry ✓, drain-split ✓)
- production native (rust)   434-440 ms  0.857-0.870×

The 36ms native→ocl_cf gap is ~97% pure-Rust-vs-ISA-L SYMBOL RATE (GZIPPY_FOLD_NODRAIN/NOCRC N=21:
drain+CRC second-touch ~0-1ms; bulk is u8-direct, banked 0f5bc85b). Advisor ×3 (synchronous): ceiling
UPHELD-W-CAVEATS twice (it corrected its own 10-20ms drain estimate after the split refuted it),
plateau VALIDATED (all STEP-2 techniques done/inapplicable; 36ms = intrinsic hand-asm-vs-LLVM gap;
table-prefetch no-headroom because the 16KiB table is L1-resident; only inline-asm transliteration
remains, highest-risk).

**SUPERVISOR GATE — FORK NOT ESCALATED.** Advisor disproof: the fork is mis-framed as engine-bound —
the engine is only ~63% of the gap (36ms); the C2 non-engine residual (ocl_cf 0.945×→rg = 21ms) is
present EVEN with real ISA-L FFI ⇒ NOT an engine problem; it's scheduling/marker-region/bootstrap =
faithful-port territory, LOWER-risk, helps BOTH goals, with a located lever
(project_confirmed_offset_prefetch_gap, ~40% of T8 wall). Escalate inline-asm ONLY when the residual is
provably engine-only (ocl_cf within noise of rg). NEXT (fresh arc, pre-register falsifier): attack C2
via parity.sh matched comparator, bounded by removal-oracle; bring a turbo-freq confirmation of the
36/21 split to the eventual escalation. NO production fix landed (measurements + byte-transparent
tooling/doc fixes only). Verdicts: plans/clean-rate-ceiling-advisor-verdict{,-recheck}.md,
plans/plateau-fork-advisor-verdict.md. Briefs: plans/clean-rate-ceiling-brief.md,
plans/fold-drain-split-result.md, plans/plateau-fork-brief.md. NO orphan processes.

## PAGE-WARMTH CLEAN ORACLE RAN → MARKER-segment warmth sub-lever REFUTED (−12% fault ceiling); CLASS not refuted; TRIAGE to engine. Advisor disproof ×3 [2026-06-07, OWNER turn, branch reimplement-isa-l, HEAD f80294ae]
Attacked the page-warmth root the prior turn named. Source-verified rg's in-place narrowing + perf-localized gzippy's fault sites + ran the rule-#3 fault-removal oracles + advisor ×3.

ROOT-CAUSE LOCALIZED (`perf record -e page-faults --call-graph fp`, symboled build, byte-exact 028bd002…cb410f):
the DOMINANT fault site is **`SegmentedU16::push_slice` = 44.52% of faults** (the u16 MARKER buffer write), NOT
the clean-tail chunk.data (decode_huffman_body_resumable 18.8%). The prior advisor verdict's "clean tail into a
separate cold chunk.data" MISLOCATED the primary term. perf stat (file sink p8): gzippy 110,617 vs rg 55,790 = 1.98×.

THREE ORACLES (locked guest, interleaved N=15, byte-exact):
- (a) GZIPPY_PREFAULT_ARENA (pre-touch+free per-worker arena): FAILED — faults UP to 193K/333K (rpmalloc handed the
  decode DIFFERENT pages; failed instrument, no counterfactual).
- (b) GZIPPY_SLAB_ALLOC (≥3 MiB resident-retain): faults −10%, matched wall WORSE 0.971× — BUT its ≥3 MiB threshold
  MISSED the 128 KiB marker segments entirely (only retained big chunk.data).
- (c) NEW GZIPPY_SLAB_THRESHOLD_KIB=64 (lowers resident-retain to capture the 128 KiB u16 marker segments; sub-MiB
  rounds to 128 KiB; FIRST oracle to target the 44% fault site): faults 110,619→97,377 (−12%, STILL 1.77× rg, FAR
  from the 55K floor); matched wall gz_s64 0.1314 vs gz_null 0.1324 = **1.008× FLAT**; rg_null 0.1132 (1.17×).

ADVISOR DISPROOF ×3 (synchronous, plans/page-warmth-rootcause-advisor-verdict.md):
- EARNED (narrow): resident-retain of u16 MARKER segments CANNOT reach rg's floor (−12% CEILING, confound-independent —
  a perfect zero-overhead version still only retains marker segments). The marker warmth sub-lever is DEAD; rely on
  the CEILING, not the flat wall.
- NOT EARNED (broad): "page-warmth refuted as a CLASS" OVERCLAIMS — the 211 MiB OUTPUT materialization (~56% of faults,
  the site rg avoids via windowed/recycled append DecodedData.hpp:344-388) was NEVER warmth-oracle'd.
- The FLAT matched wall proves NOTHING: underpowered TIE (max plausible warmth win single-digit ms is 5-10× below the
  13-18ms harness spread; rule 5/7). Drop it as evidence either way.
- rg faults half NOT only via ISA-L (engine) but via its windowed/recycled MATERIALIZATION (data structure —
  pure-Rust-portable, NOT forbidden by goal #1; = the rule-7a worker-side recycle already owed). The prior 3.26× DTLB
  regression was FRESH segmentation; RECYCLED warm-TLB windows are a DIFFERENT unrun experiment.
- rapidgzip --verbose: rg has the SAME 34.5% replaced markers (73.1M) + 0.0338s apply-window narrow ⇒ u16 footprint
  NOT gzippy-unique.

**SUPERVISOR GATE — page-warmth MARKER sub-lever dead; CLASS not refuted; NO production fix landed (3 measurement
knobs KEPT 7a, byte-exact, OFF-default, NOT promoted: GZIPPY_PREFAULT_ARENA, GZIPPY_SLAB_THRESHOLD_KIB + sub-MiB slab
granularity). NEXT (triage, advisor-endorsed): the CLEAN-ONLY ENGINE ORACLE — the 2.3× clean-rate gap ≫ warmth's
hidden few-ms, owed anyway (project_pregate_placement_is_dominant_lever); then sched slow-inject if engine bounds
short. STILL OWED for page-warmth (rule-7a convergence, the untested 56%): a recycled-window oracle on the OUTPUT
materialization (faithful DecodedData.hpp:344-388, reused 128 KiB append windows, warm-TLB).** GUEST: /tmp/gz-ft-src
build (native, sha 028bd002…cb410f OFF) + /tmp/gz-ft-sym/release/gzippy (symboled); drivers /tmp/measure_devnull.sh.
NO orphan processes. Briefs: plans/page-warmth-{rootcause,final}-brief.md.

## SUPERSEDED — OUTPUT-BINDER RECONCILIATION (matched /dev/null comparator + rg output exposure measured) → PROMPT PREMISE REFUTED: there are TWO T8 binders, NOT one; output-overlap is NON-FAITHFUL + sub-parity; the unified root = PAGE WARMTH (gzippy faults 2× rg). Advisor disproof, vendor-source-verified [2026-06-07, OWNER turn, branch reimplement-isa-l, HEAD 20084c91]
The prompt's headline ("removing gzippy's serial output writev ≈ ties rg ⇒ output is THE single T8 binder")
was a Rule-6 MISPAIRING: it compared gzippy-output-REMOVED (0.131) vs rg-output-PRESENT (0.130). Ran the
MATCHED comparator (BOTH tools output-neutralized to /dev/null) on the locked guest:

NUMBERS (locked guest REDACTED_IP, taskset 0,2,4,6,8,10,12,14, gov=perf, interleaved best-of-N, silesia
RAW=211968000, gzippy native sha 028bd002…cb410f OFF byte-exact vs rg; Battery 3 N=15, contention-paired):
gz_file 0.1677 | gz_null 0.1324 | gz_skip(null) 0.1318 | gz_ovl(file) 0.1600 | rg_file 0.1305 | rg_null 0.1148.

**RECONCILIATION (the owed split, rg's output exposure MEASURED):**
- Output exposure (file−null): gzippy **35.3ms**, rapidgzip **15.7ms** ⇒ rg ALSO pays a serial output exposure.
- Split of gzippy's 35.3ms: ~15.7ms = SHARED memory-bandwidth floor (rg pays it too, NOT a gzippy deficit);
  ~19.6ms = gzippy-SPECIFIC excess.
- **SECOND binder, survives output removal on BOTH sides: gz_null − rg_null = 17.6ms = engine+sched residual.**
  ⇒ TWO binders (output-excess + engine residual), the prompt collapsed them into one.

**ADVISOR DISPROOF (synchronous, plans/output-reconciliation-advisor-verdict.md), vendor-source-VERIFIED first-hand:**
- C1 (matched gate ⇒ two binders) UPHELD-W-CAVEATS (two MASKS of one root, not two independent levers).
- C2 (output-overlap ceiling 0.88×) REFUTED-AS-DERIVED: 0.88× is a construction, not measured; /dev/null's
  write_null skips copy_from_user so file−null bundles the SOURCE READ ⇒ the "19.6ms output excess" is largely
  the SAME cold-page tax in the 17.6ms residual — overlap (a timing change) CANNOT fix a cold-source copy. Only
  the shipped overlap writer's measured 0.81× is real; beyond is bandwidth-bound/unknown.
- C3 (rg 15.7ms = shared floor) UPHELD as a LOWER BOUND, REFUTED as "gzippy already pays it" (gzippy must EARN
  it by fixing page warmth).
- C4 (next lever = engine/page-warmth, not overlap) UPHELD (strongest claim), MECHANISM corrected.
- **KILL-SHOTS (both verified against vendor THIS turn):** (1) the "u16→u8 second pass is gzippy's deficit" is
  FALSE — rg runs the SAME applyWindow narrowing (DecodedData.hpp:164,306; MarkerVector=FasterVector<u16>);
  and rg narrows IN PLACE reusing the u16 buffer's pages (DecodedData.hpp:344-388 reinterpret_cast + reusedDataBuffers),
  gzippy decodes the clean tail into a SEPARATE chunk.data ⇒ extra allocation+faults. (2) The OVERLAP WRITER IS
  NON-FAITHFUL — rg's writeFunctor is INLINE-SYNCHRONOUS in the consumer read() loop (ParallelGzipReader.hpp:621,
  verified), NO background thread; gzippy's output_writer.rs OverlapWriter is a structural DIVERGENCE, rejectable
  under the bias guardrails regardless of ceiling. It stays OFF-by-default (experiment knob), NOT promoted to production.

**PERF EVIDENCE (root cause = page warmth):** perf stat on the CURRENT arena-wired binary —
gzippy 110,301 page-faults / 894ms task-clock vs rg 58,211 / 622ms ⇒ gzippy faults ~1.9× rg DESPITE the rpmalloc
arena already being wired (pure-rust-inflate→rpmalloc-caches→arena-allocator; U8/U16=Vec<_,RpmallocAlloc>). Faults
stay ~110K under overlap-writer AND manual-pool (topology-invariant) ⇒ intrinsic to gzippy's per-chunk allocation
FOOTPRINT (separate u16 ring + u8 contig data, cross-thread free), not pool topology. This is the unified root of
BOTH binders.

**SUPERVISOR GATE — reconciliation DONE + advisor-vetted; T8 is NOT at candidate-parity (matched 1.16×, two binders);
the output-overlap lever is REFUTED as the path (non-faithful + sub-parity ceiling). NO production fix landed this
turn (overlap writer already landed/gated last turn, KEPT OFF-default per 7a; not promoted). NEXT (faithful, gated,
do NOT start without supervisor) = the PAGE-WARMTH root: faithful in-place u16→u8 narrowing reusing the marker
buffer's pages (rg DecodedData.hpp:344-388) + cross-thread arena warmth, gated on the MATCHED gz_null-vs-rg_null
comparator (NOT skip-vs-rg-file), pre-registered falsifier. Do NOT re-open the output-overlap fork.** GUEST:
/tmp/gz-ft-src build (native, sha 028bd002…cb410f), drivers /tmp/_split_driver.sh + /tmp/measure_devnull.sh +
/root/gzippy/scripts/measure.sh. rg 0.16.0. NO orphan processes.

## SUPERSEDED — OUTPUT IS THE T8 BINDER (fulcrum_total + validated writev-removal oracle) → faithful overlap-writer LANDED (byte-exact, +3.7-6% sign-stable, rule-7a KEEP) [2026-06-07, OWNER turn, branch reimplement-isa-l]
fulcrum_total on the production T8 trace (consumer = wall-critical thread, captured to a regular tmpfs
file = measure.sh's sink; piping to sha256sum was a backpressure phantom that inflated writev 58→135ms):
consumer split = 64-67% WAIT-on-workers, **32-34% OUTPUT (consumer.writev)**, 0.8% serial compute.

CAUSAL writev-removal oracle (NEW GZIPPY_SKIP_WRITEV_SYSCALL, byte-transparent OFF==identity): gz_off
0.1653s (0.79× rg), gz_skip 0.1310s, rg 0.1311s ⇒ **removing output ≈ ties rg ⇒ OUTPUT is the single
largest T8 binder, NOT the engine.** REVISES the prior "0.135× engine table-load + 0.075× placement"
split: engine is the T1 binder (T1 skip helps only +10%, gzippy 0.615× rg), SLACK-MASKED at T8
(parallelized ~8×); the un-parallelizable serial 211 MiB output copy is the Amdahl T8 tail.

Granularity REFUTED (GZIPPY_WRITEV_CAP_KIB {2048,256,95} all TIE/worse) ⇒ lever is write TIMING not
size. Instant-feed discriminator REFUTED the advisor's "feed-rate-masked" phantom (exposure 35→31ms when
engine sped 38ms ⇒ engine-INDEPENDENT). Advisor synchronous (plans/output-binder-advisor-verdict.md):
A UPHELD-W-CAVEATS (~14ms irreducible memory-bandwidth floor rg also pays; ~20ms addressable exposure),
B REFUTED-as-stated (skip 0.98× rg not 1.0×, ~2% engine/sched residual), E REFUTED by the discriminator.
Brief: plans/output-binder-decomposition-brief.md.

LANDED (rule-7a, byte-exact 028bd002…cb410f T1+T8 stdout/pipe/file, 864 lib pass / only 2 missing-fixture
env fails + the load-flake): src/decompress/parallel/output_writer.rs — single in-order background
writer thread (GZIPPY_OVERLAP_WRITER=1, Linux+regular-fd, OFF==inline identity); consumer hands the
owned chunk to it AFTER window-publish+CRC-combine so the writev overlaps the next decode WAIT (faithful
rg writeFunctor ParallelGzipReader.hpp:521); joined+error-propagated before the trailer check. MEASURED
(5 interleaved passes): gz_overlap/gz_off 1.06/1.01/1.045/1.042/1.037× = sign-stable +3.7-6%
(0.79×→~0.81× rg), TIE-by-spread, captures ~6ms of the ~20ms ceiling (rest = shared memory-bandwidth floor).

**SUPERVISOR GATE — output localized as the T8 binder (validated removal oracle, advisor-vetted); a
faithful partial tooth banked; NEXT = the residual ~14ms memory-bandwidth floor (likely shared/irreducible),
the ~2% engine/sched residual, + the window-absent marker bootstrap. Do NOT re-open the engine table-load
fork (it's the T1 binder, slack-masked at T8).** GUEST artifacts in charter CURRENT STATE. NO orphan processes.

## SUPERSEDED — DECODE-vs-STORE LOCALIZATION + PLATEAU/FORK GATE → not separable by slow-injection; PLATEAU/FORK NOT VALIDATED, REFUTED at the engine (ocl_cf 0.925× < 1.0 ⇒ ≥0.075× of the gap is OUTSIDE the engine = placement) [2026-06-07, OWNER turn, HEAD 25846265]
Built + committed (25846265) two byte-transparent localization knobs (GZIPPY_SLOW_DECODE /
GZIPPY_SLOW_STORE) wired into the fast (VAR_V) + careful loops of decode_clean_into_contig — decode
injects after lut_litlen.decode + dist_hc.decode, store injects after literal-store +
emit_backref_contig; neither forces the careful loop (perturbation hits the PRODUCTION fast path).
Byte-exact (OFF==DEC==STORE==028bd002…cb410f x86_64 guest + arm64; 886 lib tests; net under
GZIPPY_POISON_RESERVE=1 passes bar the documented load-flake). MEASURED (locked guest, interleaved,
3 passes): both knobs monotone + on-path, dec100≈store100 (~0.81-0.83×), sleep controls survive.

ADVISOR ×2 (synchronous; plans/decode-store-localization-advisor-verdict{,-pass2}.md): PASS 1 REFUTED
"decode-compute is the more-robust binder" — the two knobs inject ADJACENT delays in ONE serial
dependency chain ⇒ they re-prove "loop on-path" but cannot separate microarch sub-resources; the
sleep discriminator is confounded by nanosleep granularity + event-count cadence. PROBES: (a) BMI2
A/B via disasm — native binary already emits 433 BZHI/PEXT (mask idioms already lower to bzhi) ⇒
bit-extraction at the BMI2 ceiling, manual PEXT/BZHI has NO headroom; (b) decode-free oracle
CONTAMINATED — wrong synthetic bytes corrupt window seeds ⇒ flip_to_clean=874, Spec other=857
re-decodes mask the wall; reverted. PASS 2 REFUTED the plateau/fork: BMI2 bounds bit-extraction NOT
the dependent TABLE-LOAD latency (unexamined); 4 authorized techniques untried (table _mm_prefetch,
static-Huffman specialization, FASTLOOP_OUTPUT_MARGIN yield-elision, single-level L1 table geometry);
DECISIVE — ocl_cf 0.925× < 1.0 ⇒ the engine fully removed is sub-parity ⇒ ≥0.075× of the gap is
OUTSIDE the engine; forking at the engine mislocates the gap.

**SUPERVISOR GATE — plateau/fork NOT validated; do NOT escalate the fork. NEXT (owed, not started):
(1) disasm a specific hot call site → FASTLOOP_OUTPUT_MARGIN yield-elision if a per-symbol branch
exists; (2) single-level L1-resident decode table (table-load-latency); (3) CORRECT-bytes replay
decode-free oracle for a validated decode-compute bound; CO-PRIMARY: the ≥0.075× non-engine
placement deficit (project_confirmed_offset_prefetch_gap).** Banked teeth UNCHANGED native_fold
~0.77-0.79× rg. GUEST artifacts in charter CURRENT STATE. NO orphan processes.

## SUPERSEDED — CORRECTNESS NET ADOPTED (poison-on, Stage-2 seam VALIDATED) + ≤0.11× RESIDUAL LOCATED on the T8 critical path (causal perturbation, advisor-UPHELD) + inner-Huffman STORE-side technique TIE'd (kept 7a) [2026-06-07, OWNER turn, HEAD 2ff19ac6]
STEP 0 (commit 24c3a04): merged test/inflate-correctness-net (seam_crossing, diff_multi_oracle,
inflate_proptest, inflate_fuzz_loop + cfg(test) 0xCD reserve-poison). Poisoned the Stage-2 contig
copy-free spare too so the FOLD copy-free clean seam is stress-tested. VALIDATED with
GZIPPY_POISON_RESERVE=1: full suite 892 pass /1 pre-existing diff_ratio flake; all 11 seam tests +
3-oracle/multi-oracle differentials + proptest + 5000-iter fuzz byte-exact. NO Stage-2 seam bug.

STEP 1a (slow-knob wire + perturbation, advisor UPHELD-WITH-CAVEATS): the Stage-2 FOLD clean tail decodes
via decode_clean_into_contig (NO ring — Stage 2 removed the clean ring-write), unperturbed by the old
ring-only GZIPPY_SLOW_MODE. Wired the clean knob into the contig loop (byte-transparent, ~24-27M hits).
CAUSAL PERTURBATION (locked guest, interleaved N=11, T8, 2 passes): off<spin50<spin100<sleep100 MONOTONIC
both passes, freq-neutral sleep preserves the delta ⇒ contig clean symbol-decode IS on the T8 critical path.
Advisor: on-path UPHELD, but a per-loop-body inject can't isolate decode-COMPUTE from store/copy bandwidth.

STEP 1b (commit 2ff19ac6, advisor UPHELD-WITH-CAVEATS): ported the ring VAR_V speculative packed
multi-literal fast loop (igzip asm:518) onto the contig clean tail, gated on sym_count>1. BYTE-EXACT
(028bd002…cb410f T1+T8 x86_64+arm64; full suite+poison+5000 fuzz green). REMOVE-AND-MEASURE (3 interleaved
passes, baseline vs fastloop2): 1.001×/1.018×/0.994× = TIE. Fast loop covers ~69% of clean events (careful
hits 27M→8.3M) ⇒ the STORE side is NOT the binding sub-resource. KEPT 7a. SUPERVISOR GATE — store technique
exhausted; binding sub-resource UNIDENTIFIED; NEXT (owed, not started) = clean-only oracle + decode-ONLY
perturbation to localize before any BMI2/packed-LUT. Banked teeth UNCHANGED; native_fold ~0.77-0.79× rg.
Briefs/verdicts: plans/contig-clean-perturbation-{brief,advisor-verdict}.md +
plans/contig-fastloop-advisor-{brief,verdict}.md. NO orphan processes.

## COPY-FREE-TO-FINAL STAGE 2 — WIRED + LANDED BYTE-EXACT + MEASURED → drain-memcpy tooth BANKED +0.05× (advisor UPHELD-WITH-CAVEATS); native_fold ~0.74×→~0.79× rg [2026-06-07, OWNER turn, HEAD 0f5bc85b]
Wired Stage-1's `decode_clean_into_contig` into the gzippy-native FOLD seam: the post-flip clean
tail now decodes DIRECTLY into chunk.data's reserved contiguous tail, DELETING the ring→chunk.data
bulk drain memcpy. FAITHFUL PREPEND (the 32 KiB window is already the contiguous tail of chunk.data
⇒ data_prefix_len stays 0, CRC-prefix-exclusion + decode_bypass landmine SIDESTEPPED; NOT the
forbidden window-in-scratch shortcut). Two synchronous advisors: pre-impl source-verify UPHELD the
key realization + landmine-sidestep + named 5 hazards; post-impl measurement UPHELD-WITH-CAVEATS.
Advisor verdict: plans/copyfree-stage2-advisor-verdict.md. **SUPERVISOR GATE — drain tooth banked
+0.05×; next = the ≤0.11× intrinsic symbol-rate inner-Huffman work bounded by ocl_cf 0.925×.**

MECHANISM (commit 0f5bc85b, +415 net over 3 files): new `MarkerStep::FlipToContig` (native,
`not(isal_clean_tail)`) → driver resumes the SAME thread-local Block in `finish_decode_chunk_contig_native`
(no ring, no drain); shared generic loop + gzippy-isal two-phase path UNCHANGED. +
`SegmentedU8::contig_decode_window` (base/cap/len re-fetched every iter, grow-safe) +
`Block::decode_clean_stored_into_contig` (post-flip stored block). Hazards: H1 release headroom
guard (contig has no ring modulo ⇒ violation = heap OOB not CRC-catchable — a real regrow-past-16MiB
bug was CAUGHT by it during testing + fixed: Vec::reserve(min_spare)); H2 stored-block; H3
commit-before-decoded_range + multi-call accounting; H4 base re-fetch; H5 native-only.

BYTE-EXACT: gzippy-native sha 028bd002…cb410f == gzip == rapidgzip on /root/silesia.gz (211968000)
T1+T8, x86_64 guest + arm64 host; gzippy-isal UNCHANGED (x86_64 Rosetta). flip_to_clean=12 (contig
route is production). 862 lib + native_fold_parity + flip-seam + 3 Stage-1 diff + 3 NEW owed-case
tests green (only pre-existing diff_ratio timing flake fails, passes isolated).

MEASURED (locked guest, interleaved measure.sh N=11/pass, sha-OK, 10 passes, A/B vs prior banked
copy-free-DRAIN baseline /tmp/gzbuild-native@9cde0b4f vs rg): stage2 strictly faster than priornative
9/10 passes; paired delta mean +0.058× median +0.044× SE±0.020 ⇒ BANKED +0.05× (advisor: drop the
+0.07 edge). Sign-stable + TRIANGULATED by the same-binary GZIPPY_FOLD_NODRAIN +0.067× (clean
isolation). Magnitude load-confounded (loadavg 2.2→5.0, autocorrelated ⇒ ~2-SE). Provenance verified:
only-delta is the Stage 2 wiring. GUEST: /tmp/gz-ft-src build (native, sha 028bd002…cb410f). NO
orphan processes.

## COPY-FREE-TO-FINAL REFACTOR — STAGE 1 LANDED byte-exact (the ~0.067× drain-memcpy tooth's hardest MECHANICAL risk retired; wall NOT yet banked, Stage 2 wiring gated) [2026-06-07, OWNER turn, HEAD c224aaad + this commit]
Two synchronous disproof advisors. The first vetted the SCOPING decision: CHECKPOINT-STAGE-1,
NOT one-pass — a gated full-wire (engine writes contiguous into chunk.data when ON, ring+drain
when OFF) CANNOT bank the wall safely this turn, because OFF==identity isolates COMMIT risk but
NOT MEASUREMENT risk: to bank you must run ON, and ON's correctness rests on the uncontained
data_prefix_len-nonzero activation + CRC-prefix-exclusion + decode_bypass serialization
round-trip; realistic one-pass outcome = a dead ON path + nothing banked (strictly worse than a
clean checkpoint). The second advisor (plans/copyfree-stage1-advisor-verdict.md) UPHELD the
LANDED Stage 1 — A (additive, zero prod caller) UPHELD, B (contig back-ref byte-equiv + proven
266-byte headroom) UPHELD, C (range-check `distance > *pos` ≡ ring's `> decoded_bytes+emitted`)
UPHELD, D (differential adequacy) UPHELD-W-CAVEATS, E (correct checkpoint, nothing to revert)
UPHELD. **SUPERVISOR GATE — Stage 1 mechanical risk retired byte-exact; Stage 2 (wire FOLD seam +
flip default + delete drain + remove-and-measure) is the next gated turn.**

LANDED (commit c224aaad, +453/-0, ONE file marker_inflate.rs, PURELY ADDITIVE, ZERO production
callers ⇒ byte-exact by construction on BOTH features + BOTH archs): (1) emit_backref_contig —
non-wrapping clean back-ref (no % U8_RING_SIZE; the 3 wrap arms collapse to word/RLE/overlap).
(2) Block::decode_clean_into_contig — clean (<false>) body → caller-supplied CONTIGUOUS buffer
with the 32 KiB window as a DICTIONARY PREFIX at base[0..window_len) (vendor setInitialWindow,
DecodedData.hpp:278-289); per-call cap to spare-(MAX_RUN_LENGTH+8), Engine-C grow-between-calls
contract. (3) 3 ring-vs-CONTIG differentials driving the REAL production read() loop, byte-equal:
window-prefix back-ref (distance==*pos==32768 → base[0]), multi-call resumable (per_call=4096),
RLE+short-distance. 3/3 pass (Rosetta x86_64 x86-64-v2); full lib 855 pass = 6 pre-existing
Rosetta/timing failures only (stash A/B == HEAD baseline); arm64 native release clean; round-trip
sha verified.

OWED FOR STAGE 2 (advisor D — do NOT assume retired): (1) stored/uncompressed clean block (contig
fn Errs; ring read() decodes it) ⇒ route/extend. (2) multi-deflate-block clean phase across a
block boundary (read_header advance with persisted *pos/decoded_bytes) UNTESTED. (3) actual
out_room-saturating REGROW UNTESTED (math present, no test drives it). PLUS the deferred landmine:
data_prefix_len=32768 activation (audit decoded_size / window-publish / consumer iovecs /
apply_window readers — FOLD has only ever run prefix==0) + CRC-prefix exclusion + decode_bypass
serialize/deserialize round-trip. FORBIDDEN shortcut: dual-region back-ref (window in scratch,
chunk.data prefix==0) — diverges from vendor prepend ⇒ violates faithful-port. Decide the FAITHFUL
PREPEND model now. RESIDUAL: banked teeth UNCHANGED at native_fold 0.737× rg; the ~0.067× tooth is
de-risked, banked by Stage 2; then ≤0.11× UPPER-BOUND intrinsic symbol rate bounded by ocl_cf
0.925×. NO orphan processes (research + 2 advisor subagents completed; no sleep sentinels).

## RING→DATA DRAIN ISOLATION RAN (fold-contig advisor's owed same-engine pure-Rust ring-copy oracle) → the 0.188× residual SPLITS: ~0.067× ring→data drain MEMCPY (recoverable free tooth) + ≤0.11× UPPER BOUND intrinsic symbol rate + ring-write; CRC is NOT a lever. NO production fix landed (banking needs the byte-exact copy-free-to-final refactor, prompt-gated) [2026-06-07, OWNER turn, HEAD 7ae5903]
Captured the gzippy-native T8 whole-system picture with the merged fulcrum_total (built native @ HEAD
fc7336c3, target-cpu=native, on /tmp/gz-ft-src), then BUILT + RAN the fold-contig advisor's owed
same-engine pure-Rust ring-copy-free isolation oracle (GZIPPY_FOLD_NODRAIN/NOCRC — measurement-only,
OFF==identity, committed 7ae5903). Synchronous disproof advisor UPHELD (C1/C3/C5 UPHELD, C2 upheld-w-
caveats; the no-op-drain is CONSERVATIVE). Charter CURRENT STATE updated. Brief:
plans/ring-drain-isolation-brief.md; verdict: plans/ring-drain-isolation-advisor-verdict.md.
**SUPERVISOR GATE — drain memcpy isolated as ~0.067× recoverable; intrinsic symbol-rate sharpened to
≤0.11× UPPER BOUND; NEXT = (1) byte-exact copy-free-to-final engine refactor to BANK ~0.067×, then
(2) the ≤0.11× inner-Huffman rate work bounded by ocl_cf 0.925×. Do NOT start without supervisor.**

fulcrum_total (native T8, trace byte-exact): routing flip_to_clean=0 finished_no_flip=16
window_seeded=2 (natural propagation, NOT the seed oracle; isal_oracle_chunks=0). The binary
routing-guard REFUSES on window_seeded>0 (advisor-flagged C4 caveat-2 brittleness — 2/16 naturally
seeded ≠ GZIPPY_SEED_WINDOWS), so read DESCRIPTIVELY with that caveat: worker.block_body 658ms SELF
(dominant = marker-engine inner decode @262 MB/s/168MB), apply_window 90ms, writev 60.8ms; consumer
61% WAIT ⇒ engine-rate-bound. The ring-write + ring→data drain are folded INSIDE block_body/drain (no
separate spans) ⇒ fulcrum_total can't split them — hence the isolation oracle.

ISOLATION (locked guest REDACTED_IP, taskset 0,2,4,6,8,10,12,14, gov=perf, interleaved measure.sh
N=11, RAW=211968000, best-of-N over 4 NON-OVERLAPPING passes, load-invariant 1.2→2.3):
native_fold ∈[0.1807,0.1831] vs nodrain ∈[0.1645,0.1684] = nodrain/native 1.083-1.109×; nodrain_nocrc
≈ nodrain; rg 0.1358-0.1464. ⇒ removing the ring→data drain memcpy moves native_fold ~0.745×→~0.812×
rg = ~+0.067× recoverable; CRC ≈ free. SELF-TEST: OFF==rg==028bd002…cb410f (x86_64 guest + arm64);
NODRAIN sha differs (fires), exits 1 only at the terminal trailer CRC32 (full decode ran first). 857
lib tests green.

ADVISOR: C1 UPHELD-W-CAVEATS (best-of-N on one-sided noise = correct estimator; interleaved +
load-invariant = freq-neutral; small turbo component not fully excluded w/o pinned-freq pass). C2
UPHELD-W-CAVEATS (≤0.11× still confounds ring-write + ISA-L-vs-pure-Rust engine diff = UPPER BOUND on
symbol rate; safe unconditional = intrinsic ≤0.188×). C3 UPHELD (CRC reads ring slice, drain-
independent). C5 UPHELD (copy-free-to-final is FAITHFUL: vendor decodes clean BULK to contiguous u8
with no u16 ring, DecodedData.hpp:278-289 — gzippy's clean-phase drain has no vendor counterpart;
nuance: vendor concatenates at merge, so claim "no clean-phase ring/narrow" not "zero copies"). KEY:
no-op-drain is CONSERVATIVE (cold chunk.data reads in writev + window-publish cost nodrain EXTRA) ⇒
true drain ≥ +0.067×.

NEXT (gate, do NOT start): (1) BANK ~0.067× — pure-Rust clean (`<false>`) phase writes u8 DIRECTLY
into chunk.data's reserved tail (reuse writable_tail_reserve+commit), back-refs from that tail, NO
output_ring for the clean phase; non-trivial engine rewrite (clean-phase addressing + back-ref resolve
+ flip-seam + >16MiB reserve-clamp fallback), correctness-sensitive at the flip seam; bound by the
nodrain knob, byte-exact dual-sha, remove-and-measure. (2) THEN ≤0.11× inner-Huffman rate (BMI2 PEXT/
BZHI, wider multi-literal, ISA-L-class packed-u32 LUT) bounded by ocl_cf 0.925×, never the VAR_VI slope.
GUEST: /tmp/gz-ft-src (source @fc7336c3 + knobs; symlinks vendor→/root/gzippy/vendor, /tmp/fulcrum→
/root/fulcrum), build /tmp/gz-ft-src/target/release/gzippy (native, sha 028bd002…cb410f OFF). Driver
/tmp/isolation_wall.sh. Trace /tmp/ft-art/. NO orphan processes.

## SUPERSEDED — CADENCE/INTRINSIC SPLIT RUN (advisor's owed symmetric control) → CADENCE TAX IS REAL + RECOVERED + LANDED: copy-free FOLD clean drain, native_fold 0.678× → 0.737× rg = +0.059× banked ratchet tooth; residual ≤0.188× UPPER BOUND on intrinsic rate [2026-06-07, OWNER turn, HEAD 9cde0b4f]
Ran the symmetric control (give the gzippy-native FOLD `<false>` clean tail one contiguous output
region, no per-block refetch/grow). The FIRST control (pending_clean-only) TIE'd — disproof advisor
caught I'd left the DOMINANT per-block ring→u8buf copy (fresh Vec::with_capacity + byte loop in
marker_inflate::drain_to_output) untouched. Ran the advisor's owed copy-free-drain control: it MOVED
the wall, proving a free cadence component was mis-booked as intrinsic. LANDED the fix (commit 9cde0b4f,
byte-exact). Synchronous disproof advisor ×2 (PASS 1 REFUTED under-scoped C1; PASS 2 CONFIRMED the
landed fix, 3 corrections applied). Charter CURRENT STATE updated. Advisors:
plans/fold-contig-split-advisor-verdict.md + plans/fold-contig-landed-advisor-verdict.md.
**SUPERVISOR GATE — cadence tax RECOVERED + banked (+0.059×); next = inner-Huffman RATE work on the
`<false>` clean path, bounded by ocl_cf 0.925× (engine-removed), never the VAR_VI slope.**

THE FIX (production default, byte-exact, advisor-CONFIRMED): marker_inflate::drain_to_output clean
branch pushes ≤2 CONTIGUOUS u8 ring slices straight to the sink (no per-block u8buf alloc + byte loop);
new ContigFoldSink (replaced+DELETED UnifiedMarkerSink) writes them DIRECTLY into a pre-reserved
contiguous chunk.data (no pending_clean middle-man, no second append_clean copy, no regrow);
ChunkData::reserve_clean (clamped 16 MiB).

SELF-TEST (Rule 4): rg == native_FINAL == isal_FINAL sha 028bd002…cb410f (byte-exact, OFF==identity);
flip_to_clean=0 finished_no_flip=16 (FOLD preserved, window-absent bootstrap preserved, did NOT seed);
857 lib + native_fold_parity + flip-seam differentials green (Rosetta x86_64 + arm64).

THE SPLIT (locked guest, interleaved measure.sh, sha-OK, 6 passes): old(triple-buffer) 0.634× <
new_off(copy-free drain) 0.674× < new_contig(fully copy-free) 0.717× rg means, MONOTONIC every pass.
copy#1 (ring→u8buf) +0.040×, copy#2/3/grow +0.043×. BANKED (quiet box, default binary, 3-pass):
native_fold 0.678× → 0.737× rg = +0.059× (the +0.083× split-sum is load-inflated, sign/monotonicity
evidence only). isal_prod also improved to ~0.80× (shares the copy-free drain in its marker bootstrap).

ADVISOR: L1 mechanism CONFIRMED (copy-free drain + contig sink, byte-exact even on the ring wrap split);
L1 magnitude OVERSTATED → banked is +0.059× not +0.083×; L2 (residual = intrinsic symbol rate) NOT
LICENSED → ≤0.188× is an UPPER BOUND (ocl_cf is ring-free + copy-free-to-final + a different engine, so
it confounds symbol rate + ring-write + ring→data drain memcpy); L3 (ContigFoldSink default + delete
UnifiedMarkerSink) CONFIRMED (no live caller, two-phase CleanTailSink untouched, sink overrides correct,
blast radius bounded). 3 corrections (re-label +0.059×, residual upper bound, reserve heuristic + clamp)
ALL APPLIED.

NEXT (gate, do NOT start): inner-Huffman RATE on the `<false>` clean path (BMI2 PEXT/BZHI, wider
multi-literal, ISA-L-class packed-u32 LUT — CLAUDE.md-authorized), bounded by ocl_cf 0.925×, never the
slope. A clean intrinsic-rate isolation needs a same-engine pure-Rust ring-based copy-free-to-final
oracle (does not exist yet) to strip the residual ring-drain memcpy from the 0.188×. GUEST:
/tmp/gzbuild-native (FINAL) + /tmp/gzbuild-head (FINAL) + /tmp/gzippy-old-drain (pre-fix baseline), all
relevant sha 028bd002…cb410f; drivers /tmp/{baseline,contig,contig2,split}_wall.sh + contig_selftest.sh
(bash). NO orphan processes.

## SUPERSEDED — COPY-FREE CLEAN-TAIL ORACLE RAN → prior INCONCLUSIVE REVERSED: clean-decode subsystem IS a wall lever (copy-free ISA-L 0.87–0.925× TIE vs pure-Rust 0.73–0.755×). UNIFIED single-primitive engine ALREADY EXISTS (gzippy-native FOLD, byte-exact) but is the SLOWEST pure-Rust at the wall (0.68×); next lever bounded < 0.14× = cadence-tax + intrinsic clean rate [2026-06-07, OWNER turn, HEAD 7aae6c4a + copy-free overlay, measured /tmp/gzbuild-head + /tmp/gzbuild-native]
Made the clean-tail oracle COPY-FREE (the advisor-OWED fix): new `decompress_deflate_from_bit_into`
(ISA-L decodes DIRECTLY into the chunk buffer — no 64 MiB Vec, no copy) + `writable_tail_reserve` +
`decoded_range`. The copy-free ISA-L clean tail (unseeded, window-absent preserved) hits 0.87–0.925× rg
= TIE vs production pure-Rust 0.73–0.755× — Δ ~0.14× ≫ spread, sign-stable ×3, freq-neutral. Then
source-verified + measured that the UNIFIED single-primitive engine ALREADY EXISTS as gzippy-native (the
FOLD: marker_inflate Engine M flips in-place via `flip_repack_to_u8` = vendor setInitialWindow and
continues the clean tail on the SAME cursor — the governing one-engine memory), byte-exact sha
028bd002…cb410f, flip_to_clean=0 finished_no_flip=16. The unified PURE-RUST engine is the SLOWEST at the
wall (native_fold 0.68× rg) — marker_inflate `<false>` clean rate trails resumable AND ISA-L. Synchronous
disproof advisor (plans/copy-free-oracle-advisor-verdict.md) UPHELD-WITH-CAVEATS. Charter CURRENT STATE
updated. Brief: plans/copy-free-oracle-brief.md.
**SUPERVISOR GATE — clean-engine question SETTLED (subsystem is a lever, ceiling readable = 0.925×); the
unified-engine PORT is DONE structurally (gzippy-native is the production unified path); NO production fix
landed (oracle measurement-only, byte-transparent); next = the clean-RATE fix on the unified engine,
scoped + bounded.**

SELF-TEST (Rule 4): PROD == copy-free ORACLE == rg sha 028bd002…cb410f (byte-exact, OFF==identity);
isal_oracle_chunks=14 isal_oracle_fallbacks=0; routing = prod unseeded (window-absent preserved). 877 lib
+ seam + 10 segmented_buffer tests green (Rosetta x86_64).

WALL (locked guest, interleaved N=11, sha-OK, gov=perf): rg 1.000 | ocl_cf (copy-free ISA-L unified)
0.895/0.892/0.870/0.925/0.905× = TIE | isal_prod (two-engine, pure-Rust clean) 0.735/0.755/0.746/0.733× |
native_fold (unified pure-Rust FOLD) 0.685/0.676× = SLOWEST.

ADVISOR: C1 UPHELD-WITH-CAVEATS (subsystem is a real lever, Δ≫spread, copy gone, reverses INCONCLUSIVE;
but 0.14× bundles ISA-L symbol rate + per-128-KiB resumable yield/refetch tax + one-shot reserve vs grow —
"RATE is the lever" over-attributes). C2 UPHELD (residual = window-absent structure). C3 DIRECTION UPHELD
(unify right), COROLLARY REFUTED (pure-Rust gap to close < 0.14×; cadence+grow are recoverable WITHOUT
raising symbol rate; VAR_VI 0.6× does NOT bound a tax-shedding primitive). MOST ACTIONABLE: run the
symmetric control — give the PURE-RUST `<false>` clean tail one large contiguous window (no per-128-KiB
refetch/yield) and re-measure native_fold; recovered portion = cadence/grow (free), remainder = intrinsic
rate that sets the real no-FFI 1.0× bar.

NEXT (gate, do NOT start): the CLEAN-RATE fix on gzippy-native FOLD. (1) symmetric control = large
contiguous window for the `<false>` tail; (2) close the intrinsic-rate remainder (BMI2/multi-literal/
packed-u32 LUT) up to the < 0.14× bound; ceiling = copy-free ocl_cf 0.925× (engine-removed), never the
slope. GUEST: /tmp/gzbuild-head (isal+copy-free oracle) + /tmp/gzbuild-native (FOLD), both sha
028bd002…cb410f; drivers /tmp/oracle_{selftest,wall}.sh (bash); 3 modified files overlaid on guest tree.
NO orphan processes.

## SUPERSEDED — DECISIVE WALL ORACLE RAN (window-absent-PRESERVING ISA-L clean-tail removal) → clean-engine binder INCONCLUSIVE (instrument contaminated by its own copy); RECONCILIATION SOLID (gap = window-absent STRUCTURE, not the clean engine); NO FORK established [2026-06-07, OWNER turn, HEAD 7aae6c4a, measured /tmp/gzbuild-head]
Ran the decisive WALL-level removal oracle the prior advisor OWED: `GZIPPY_ISAL_ENGINE_ORACLE=1`
UNSEEDED (window-absent routing preserved per the charter OSCILLATION rule) replaces ONLY the
post-flip clean tail with real ISA-L. Plus the seedfull↔production reconciliation (engine held
constant). Synchronous disproof advisor REFUTED my Conclusion-1 framing. Charter CURRENT STATE
updated. Advisor verdict: plans/clean-tail-wall-oracle-advisor-verdict.md. Brief:
plans/clean-tail-wall-oracle-brief.md. Pre-reg: plans/clean-tail-wall-oracle-prereg.md.
**SUPERVISOR GATE — clean-engine binder UNDECIDED (copy-free oracle owed); NO fix landed; NO
fork escalated; the user-constraint fork is NOT forced (advisor: no fork — unify the two engines).**

SELF-TEST (Rule 4): OFF==ON==rg sha 028bd002…cb410f (byte-exact); isal_oracle_chunks=14
isal_oracle_fallbacks=0; routing flip_to_clean=12 finished_no_flip=4 window_seeded=2 = prod
unseeded ⇒ 89% window-absent bootstrap preserved.

RESULT 1 — WALL (3 interleaved N=11, sha-OK, locked guest, gov=perf): prod 0.744/0.754/0.755× rg;
ocl (ISA-L clean tail, unseeded) 0.698/0.686/0.702× rg. The ISA-L clean-tail oracle did NOT beat
production. ADVISOR (load-bearing, CORRECT): INCONCLUSIVE — the oracle's per-chunk 64 MiB
alloc+to_vec copy (gzip_chunk.rs:203,247-256) costs ~0.17× (= the prod→ocl gap AND the
gap-to-threshold); my own model implies S≈C≈0.17×, i.e. the clean engine may SAVE ~0.17× of wall
exactly masked by the copy. A handicapped contender failing to win is uninformative about a
speed-UP ceiling (Rule 3). CANNOT declare engine slack NOR engine binder.

RESULT 2 — RECONCILIATION (UPHELD, engine constant = ISA-L both, 2 runs): ocl_unseed (bootstrap
preserved) 0.697/0.701× vs ocl_seed (seeded, no bootstrap) 0.860/0.857× rg. The seedfull↔production
gap is the WINDOW-ABSENT STRUCTURE (~0.16× of wall), NOT the clean engine. CAVEAT: do NOT
sub-credit to marker RATE specifically — seeding also removes 13→0 spec-fail re-decodes
(project_confirmed_offset_prefetch_gap) + flip machinery; the 0.120s≈0.322s match is the SUM-vs-wall
trap. Structure-level cause solid; rate-vs-spec-fail split NOT isolated.

Q4 NO FORK (advisor, vendor deflate.hpp:1277/1285/1452-1453/1585-1666/1600): rg uses ONE
width-templated `readInternalCompressedMultiCached` for BOTH u16 markers and u8 clean; gzippy's
two-engine split (marker_inflate u16 + resumable u8) is the divergence. The marker rate is likely
bounded by the SAME primitive ceiling as the clean (VAR_VI 0.6×). The faithful move is to COLLAPSE
marker+clean into ONE primitive ([[project_faithful_unified_decoder_over_perf]]) — not a fork.

NEXT (gate, do NOT start): (1) make the oracle copy-free (decode ISA-L into writable_tail) so the
clean-engine ceiling becomes readable; THEN (2) the UNIFIED single-primitive engine. GUEST:
/tmp/gzbuild-head; drivers /tmp/oracle_{selftest,wall,reconcile,verbose}.sh (bash); seeds
/tmp/seeds.bin. NO orphan processes.

## SUPERSEDED — 1.6× WINDOW-ABSENT decodeBlock GAP ATTRIBUTED → it's the CLEAN u8 TAIL decoder (pure-Rust ~2.3× slower than ISA-L), NOT the u16 marker loop [2026-06-07, OWNER turn, HEAD 7aae6c4a + overlay]
Source-diffed the window-absent decode path vs vendor, ran apples-to-apples --verbose
(gzippy + rg, T8 unseeded, locked guest) + a causal SLOW_MODE A/B (freq-neutral),
synchronous disproof advisor. STOPPED at attribution + scoped fix (NO fix landed, per
prompt). Charter CURRENT STATE updated. Advisor: plans/window-absent-attribution-advisor-verdict.md.
Attribution: plans/window-absent-attribution.md. Brief: plans/window-absent-attribution-brief.md.
**SUPERVISOR GATE — attribution advisor-vetted; NO fix landed; next = run the Phase-0
ISA-L removal oracle on the FlipToClean clean tail to bound the wall payoff (Rule 3),
then the scoped clean-engine fix.**

NUMBERS (T8 unseeded, locked guest REDACTED_IP, taskset 0,2,4,6,8,10,12,14, gov=perf):
gzippy decodeBlock SUM 0.805-0.831s vs rg 0.4995s = 1.61-1.66×. Per-engine: gzippy
marker bootstrap body 0.323s (73.0M @ 226-235 MB/s) vs rg "custom inflate" 0.4748s =
gzippy 0.68× FASTER; gzippy clean tail ≈0.48s (≈139M @ ≈290 MB/s, subtraction) vs rg
"ISA-L" 0.2065s = ≈2.3× SLOWER. Routing: flip_to_clean=12 finished_no_flip=4
window_seeded=2.

CAUSAL (ΔdecodeBlock SUM, freq-neutral sleep control, baseline 0.831s): MARKER+100%
→0.965 (+134ms); CLEAN+100% →1.025 (+194ms, marker body UNCHANGED). Sleep: marker +142,
clean +248. CLEAN inject lands in resumable.rs:1199 (the post-flip clean engine), marker
body untouched, yet adds the LARGER ΔdecodeBlock ⇒ the clean tail is the dominant
decodeBlock term, NOT the marker loop.

ROOT CAUSE (source): flip threshold byte-identical to vendor (marker_inflate.rs:1116-1119
↔ deflate.hpp:1282-1284). Two-phase routing (gzip_chunk.rs:1397-1410, isal_clean_tail):
Engine M u16 bootstrap → FlipToClean → Engine C = StreamingInflateWrapper =
unified::Inflate<Clean> = pure-Rust resumable.rs. The measured gzippy-ISAL build's clean
tail is PURE-RUST, NOT ISA-L FFI (resumable.rs:1182-1192; build.rs:98 "REAL ISA-L FFI"
comment was STALE — FIXED, comment-only/byte-transparent). rg WITH_ISAL uses real ISA-L
(deflate.hpp:1452-1453). So the 1.6× = gzippy's pure-Rust clean engine (~0.6× ISA-L
primitive plateau) decoding the clean BULK ~2.3× slower than ISA-L.

ATTRIBUTION among prompt's 3 candidates: (a) marker inner loop REFUTED-as-prime (gzippy
marker FASTER than rg); (b) u16-over-clean-bulk REFUTED (flip byte-identical; post-flip
bulk is u8); (c) table-build REFUTED (shared, seedfull ties). CAUSE = pure-Rust clean
tail slower than ISA-L.

ADVISOR: core UPHELD (causally airtight); CONFIRMED the build.rs stale-comment risk;
REFUTED "actionable now" (Δwall payoff needs the Phase-0 ISA-L removal oracle, Rule 3,
SUM slack-masked at Fill 85%); 0.68×/2.3× are subtraction hedges (proven claim = "clean
tail is the bigger decodeBlock term"); candidate-1 ISA-L FFI = goal #2 (re-adds FFI),
candidate-2 faster pure-Rust clean engine = goal #1 advance.

SCOPED FIX (gate, do NOT start): (1) goal#2 route FlipToClean clean tail through ISA-L
FFI (= rg WITH_ISAL); (2) goal#1 faster pure-Rust clean u8 engine (BMI2/multi-literal/
ISA-L-LUT). Bound with GZIPPY_ISAL_ENGINE_ORACLE on the clean tail (FALLBACKS==0) FIRST.
GUEST: /tmp/gzbuild-head/release/gzippy (measured, clean HEAD isal native); rg 0.16.0;
captures /tmp/{rg_verbose,gz_verbose,m100,c100,cs,ms}.txt. NO orphan processes.

## CORRECTED overlap oracle RUN → OVERLAP DEAD AS A LEVER; all 3 scheduling levers causally exhausted; binder RELOCATED to the marker-engine decode rate [2026-06-07, OWNER turn, HEAD 7aae6c4a + corrected-oracle overlay]
Corrected the backwards overlap oracle (warm-all-then-drain → real in-flight
overlap dispatch via `submit_prefetch`, non-blocking), measured the registered
decider on the locked guest, removed the advisor-flagged 4096 retention confound,
and tested the two remaining scheduling sub-levers. Two synchronous disproof-advisor
passes (PASS 2 reversed PASS 1's own "F1 likely holds via overlap"). Charter CURRENT
STATE updated. Advisor: plans/corrected-overlap-advisor-verdict.md (+FOLLOW-UP
VERDICT section). Brief: plans/corrected-overlap-advisor-brief.md.
**SUPERVISOR GATE — overlap REFUTED, engine relocation advisor-vetted (direction),
NO engine fix landed; next loop = faithful engine direction (u8-direct clean path).**

THE CORRECTION (perfect_overlap.rs + chunk_fetcher.rs perfect_overlap_warm): dispatch
EVERY chunk's decode as an IN-FLIGHT prefetch up-front (`submit_prefetch(part_key,rx)`
= vendor m_prefetching.emplace, BlockFetcher.hpp:558), NON-BLOCKING, return immediately
so the in-order consumer_loop runs CONCURRENTLY with the still-running decodes = real
decode↔drain overlap. Removed the 4096 cache-cap bump (retention confound).

SELF-TEST (Rule 4): OFF==identity AND ON byte-exact sha 028bd002…cb410f on arm64-native
(local) + x86_64 gzippy-isal native (guest); path=ParallelSM; dispatch phase 0.0007s
(non-blocking); warm_hit_frac 0.882 (15/17, 2 offset-0 misses).

DECIDER NUMBER (T8, measure.sh interleaved N=11, sha-OK, locked guest, gov=perf, 5 runs):
perfovl CORRECTED+retention-fixed = 0.187-0.192s = 0.684-0.695× rg vs HEAD 0.174-0.177s
= 0.730-0.754× (rg 0.130s). **The corrected overlap oracle is sign-stably ~5-7% SLOWER
than production — does NOT reach the tie; retention fix did NOT rescue it.**

RESOLVE-AHEAD SATURATED (verbose): Worker resolve-ahead ok=13/13 (head), 14/14 (perfovl)
= drain-hiding LIVE on the production path ~82% coverage; std::future::get is a wait on
the DECODE future, not resolve (resolve runs earlier on the pool). Drain already hidden.

FINER-CHUNKING REFUTED (GZIPPY_CHUNK_KIB sweep, byte-exact, interleaved vs rg, 2 runs):
k4096(17 chunks)≈k2048(34)≈k1024(68) all 0.68-0.72× rg, FLAT-to-WORSE. decodeBlock SUM
stays ~1.1s, Fill DROPS 87%→77% (per-chunk bootstrap overhead cancels the tail-wave gain).
The advisor's ~0.04s tail-wave-quantization hypothesis did NOT materialize.

ADVISOR ×2: PASS 1 REFUTED my decider C1 (retention confound + the real lever is
resolve-ahead), said F1 likely holds via overlap. PASS 2 (after resolve-ahead-saturated
+ retention-fixed + finer-chunking evidence) REVERSED: F1-via-overlap/drain REFUTED;
binder relocates to the per-thread decode floor (engine) — DIRECTION upheld, 1.6×
magnitude UNVERIFIED; target the u8-DIRECT clean path (governing memory), not just a
faster u16 ring.

CONCLUSION (causal, advisor-vetted): the T8 wall is NOT scheduling-bound. All three
scheduling levers exhausted (dispatch-depth null/worse, drain-hiding saturated, tail-wave
flat-to-worse). Residual ~0.70× gap = per-thread MARKER-ENGINE decode rate. NEXT (gate):
source-diff window-absent decode_chunk_unified_marker vs vendor to attribute the gap
(marker inner loop / u16-vs-u8 width / table-build), faithful target = u8-direct clean
path + readInternalCompressedMultiCached marker port; bound any engine fix with the
Phase-0 ISA-L engine-removal oracle (already ties seeded), never the slope.
GUEST: src overlaid to HEAD+corrected-oracle, build /tmp/gzbuild-po (sha 028bd002…cb410f),
drivers /tmp/po_measure.sh + /tmp/chunk_sweep.sh. NO orphan processes.

## SUPERSEDED — GZIPPY_PERFECT_OVERLAP (the registered decider, NEVER-BEFORE-RUN) BUILT + SELF-TESTED + RUN; advisor REFUTED my read (oracle built BACKWARDS) [2026-06-07, OWNER turn, HEAD 7aae6c4a + oracle overlay]
Closed the live Rule-3 violation flagged by the supervisor coach (PROCESS FIX #1):
the registered decider `GZIPPY_PERFECT_OVERLAP` had NEVER been run. Built it
byte-exact (src/decompress/parallel/perfect_overlap.rs + perfect_overlap_warm in
chunk_fetcher.rs), self-tested (Rule 4), measured on the locked guest, ran the
synchronous disproof advisor. Advisor caught a LOAD-BEARING error. Charter CURRENT
STATE + prereg RESOLUTION #2 updated. Advisor: plans/perfect-overlap-advisor-verdict.md.
Brief: plans/perfect-overlap-advisor-brief.md. **SUPERVISOR GATE — decider RUN, but
F1 UNDECIDED (my oracle tested the wrong schedule); do NOT declare STOP/TIE.**

DECIDER NUMBER (lead with the causal oracle, T8, measure.sh interleaved N=11, sha-OK,
2 runs): perfovl 0.225-0.227s = 0.581-0.583× rg; HEAD 0.177s = 0.740×; rg 0.131s.
**The oracle is SLOWER than production.**

WHY (advisor, load-bearing): my oracle runs warm (decode-ALL) FULLY then drain
(resolve-chain+write) — it SERIALIZES the two phases production already OVERLAPS =
an ANTI-overlap. Its wall is a pessimistic SUM (warm 0.117 + drain 0.066). An
upper bound built by DESTROYING overlap cannot falsify F1 (the TIE claim) — the
symmetric of last turn's upper-bound-can't-fire-F2. My C1/C3/C4 REFUTED.

GENUINE FINDING (advisor-corrected): warm alone = 0.117s is a TRUE LOWER bound on
any schedule's wall (all chunks must decode; drain 0.066 < warm hides under it).
**0.117 < rg WALL 0.131 < tie threshold 0.138** = INSIDE the tie zone. I mis-read it
"above the tie" only by comparing 0.117 to rg's decode FLOOR (0.085) not rg's WALL
(0.131). Matched floor-to-floor = warm 0.117 vs rg Real Decode 0.104 = 1.13×. **Read
correctly: the T8 TIE IS REACHABLE by better decode↔drain OVERLAP — the scheduling
direction is NOT refuted; this oracle FAILED TO TEST it.**

SELF-TEST (Rule 4): sha 028bd002…cb410f byte-identical w/ and w/o the oracle on
arm64 + x86_64 guest; warm_hit_frac 0.88-0.96 (2 misses = offset-0 stream-start).
Byte-transparent; the warm cache really removed the head-of-line wait.

STILL OPEN: can a REAL OVERLAPPED schedule collapse production 0.177 toward the
0.117-0.13 floor? Needs a CORRECTED oracle (warm OVERLAPPING drain, pipelined
per-chunk as predecessor windows land), NOT warm-all-then-drain. F1 UNDECIDED.
SCOPED NEXT (gate): build the corrected overlap oracle to decide F1, OR (lower bound
already says reachable) go to the faithful decode↔drain overlap fix
(project_confirmed_offset_prefetch_gap dispatch-TIMING) bounded by that corrected
oracle first. GUEST: src overlaid to HEAD+oracle, build /tmp/gzbuild-po, driver
/tmp/po_measure.sh. NO orphan processes.

## SUPERSEDED — COUNTER RENAMED (anti-inversion) + SCHEDULING/SERIAL CEILING BOUNDED via real oracles; advisor REFUTED my arithmetic F2 over-reach [2026-06-07, OWNER turn, HEAD f1aceee1]
Renamed the inversion-prone counter byte-exact (commit f1aceee1:
`BOOTSTRAP_POST_FLIP_U16_BYTES` → `BOOTSTRAP_CLEAN_FLIPPED_BYTES`; it counts the
marker-FREE CLEAN-flipped complement, NOT marker-ring bytes — the exact inversion
that bit C3 repeatedly; verbose label now self-documents). Then bounded the
scheduling/serial ceiling with REAL removal oracles on the locked guest (src rsynced
to HEAD, /tmp/gzbuild-head, sha 028bd002…cb410f every cell). Charter CURRENT STATE +
prereg updated. Advisor: plans/scheduling-ceiling-advisor-verdict.md. Pre-reg:
plans/scheduling-ceiling-prereg.md. Brief: plans/scheduling-ceiling-advisor-brief.md.
**SUPERVISOR GATE — ceiling bounded, NO engine fix landed (binder coupled/unconfirmed).**

NUMBERS (T8, interleaved measure.sh, sha-exact): HEAD 0.174-0.177s = 0.736-0.755× rg
(rg 0.130s). seedfull (GZIPPY_SEED_WINDOWS = the faithful perfect-window-overlap oracle)
0.128s = 1.029× = TIE; T16 0.128s = 1.121× WIN. NO_PREFETCH negative control 0.523s =
0.253× (3× SLOWER ⇒ scheduling firmly critical). Verbose: gzippy decodeBlock SUM 0.803
vs rg 0.502 (1.6×); Real Decode 0.116 vs 0.084; future::get 0.089 (T16 0.046, HALVES).

KEY (oracle-grounded, advisor-corrected): the T8 TIE IS reachable (seedfull, F1) BUT
the scheduling overlap AND the window-absent marker-engine rate are LIVE + COUPLED
(window-present ⇒ CLEAN engine, gzip_chunk.rs:790 vs :826) — neither isolable. My
arithmetic "engine binds (F2)" was a forbidden Rule-3 extrapolation (the 0.116+0.043
sum was a STRICT UPPER BOUND, double-counting the overlapping tail) — advisor REFUTED
C2 (scheduling IS critical: future::get halving = criticality, NO_PREFETCH 3×) and C3
(F2). C1 (engine reaches wall) UPHELD-WITH-CAVEATS. C4 (backward marker scan) UNCONFIRMED/
implausible (flip-to-clean at 32KiB confines it). rg's mechanism source-verified
(GzipChunkFetcher.hpp:479/:513/:559 — main-thread uncompressed last-window publish on
the named serial critical path; gzippy ALREADY ports it) ⇒ residual is dispatch TIMING,
not a missing mechanism nor horizon DEPTH (vendor-identical). SCOPED NEXT (gate, must
CAUSALLY PERTURB FIRST): (a) faster window-absent marker engine OR (b) earlier window
publish so more chunks hit clean at high T (project_confirmed_offset_prefetch_gap). NO
orphan processes (advisor wrapper+sleep killed; guest clean).

## MARKER FAST LOOP LANDED → rg's multi-cached u16 marker loop ported; T8 wall = TIE (no move), byte-exact, KEPT per 7a. Advisor C1 UPHELD / C3 REFUTED my mechanism [2026-06-07, OWNER turn, HEAD 04fda86d]
Ported step (i) of the bounded plan: rg's multi-cached u16 marker FAST LOOP
(vendor `readInternalCompressedMultiCached` deflate.hpp:1585-1666). Added a
speculative software-pipelined fast loop to the u16 MARKER path
(`read_internal_compressed_specialized::<true>`, marker_inflate.rs `'mfast` loop),
mirroring the clean path. rg runs the SAME tight multi-cached loop for u16 markers
as u8 clean; gzippy's marker path was stuck on the careful per-symbol loop. Three
faithful u16 deltas (widened-u16 speculative store, distance_marker += lit_prefix +
emit_backref_ring::<true>, no marker-window range check). Charter CURRENT STATE
updated. Advisor: plans/marker-loop-port-advisor-verdict.md. Brief:
plans/marker-loop-port-brief.md.

BYTE-EXACT: gzippy-native arm64 + gzippy-isal guest x86_64, sha 028bd002…cb410f
T1/T8/T16 path=ParallelSM. 856 lib tests (1 fail = pre-existing flaky diff_ratio).
Seam + native_fold_parity green.

REMOVE-AND-MEASURE (locked guest REDACTED_IP double-ssh, 16c gov=perf turbo-on,
taskset 0,2,4,6,8,10,12,14, T8, measure.sh interleaved N=11, RAW=68229982, sha-OK):
markerfast vs mergefix(prior HEAD) = +1.2% / +3.0% / +0.0% over 3 runs = TIE
(spread 10-38%). decodeBlock 0.9568→0.9485s (~0.9%); rg decodeBlock 0.500s ⇒ gzippy
still ~1.9× but SLACK-MASKED (Fill 87%, Total Real Decode 0.137s, wall 0.175s =
0.73× rg, unchanged). KEPT per rule 7a (faithful TIE, gain latent behind binder).

ADVISOR: C1 UPHELD (widening + wrap-safety + no-desync + faithful range-check drop
all source-verified; emit_backref_ring is one shared fn). C2 UPHELD-WITH-CAVEATS
(honest TIE + interleaved freq-neutral, but spread can't see ≤10-20%). C3 REFUTED
(LOAD-BEARING): I read BOOTSTRAP_POST_FLIP_U16_BYTES BACKWARDS — it counts CLEAN
flipped bytes (gzip_chunk.rs:1489), so 2.0% is the sliver the loop does NOT touch;
the loop owns the ~98% marker COMPLEMENT. The exact counter-inversion the charter
warns about. Commit msg corrected.

CORRECTED PREMISE: "decodeBlock 1.69× = the marker loop" is now suspect — the loop
owns ~98% of bootstrap body yet barely moved decodeBlock and did NOT move the wall.
The engine SUM (~1.9×) is real but SLACK-MASKED at Fill 87% (Phase-0 already showed
engine TIE-vs-TIE seeded). The T8 binder is NOT the per-thread engine compute. NEXT
(do NOT start — supervisor gate): re-perceive the wall; the gap is the SCHEDULING/
SERIAL term (pool-fill + in-order consumer head-of-line wait =
project_confirmed_offset_prefetch_gap) — bound THAT with a removal oracle, not the
slack-masked engine. GUEST: /root/gzippy tree RESTORED to baseline (marker patch
reversed) + mergefix overlay; builds /tmp/gzbuild-{base,mergefix,markerfast} (all
sha 028bd002…cb410f); drivers /tmp/markerfast_{measure,trace}.sh + /tmp/sha_markerfast.sh
(bash); patch /tmp/marker_fastloop.patch. NO orphan processes.

## MERGE-REMOVAL LANDED → rg's view-based applyWindow ported; T8 wall MOVED +12% (0.65×→0.73× rg), byte-exact, advisor-UPHELD [2026-06-07, OWNER turn, branch reimplement-isa-l]
Executed step 1 of the bounded plan: port (ii) rg's view-based applyWindow = drop the redundant
full-output memcpy in `merge_resolved_markers_into_data`. Charter CURRENT STATE updated. Advisor:
plans/merge-removal-advisor-verdict.md. Brief: plans/merge-removal-advisor-brief.md.

CHANGE (chunk_fetcher.rs:2453 resolve_chunk_markers_on_chunk + chunk_data.rs): DROP
`merge_resolved_markers_into_data()` (~68MB full-output alloc+memcpy, segmented_buffer.rs:356) AND
the eager `recycle_markers_after_resolution()`. Narrowed marker bytes stay in `data_with_markers`
(narrowed_len set), emitted zero-copy via append_narrowed_iovecs; recycle DEFERRED behind the
consumer writev (defer_chunk_recycle → recycle_decoded_buffers frees both buffers). `contains_markers`
treats narrowed_len>0 as resolved (post-narrow u16 high bytes are stale). populate_subchunk_windows
assert relaxed. + debug-only double-resolve tripwire (advisor rec, byte-transparent). New test
populate_subchunk_windows_unmerged_view_based_apply_window.

BYTE-EXACT: gzippy-isal native (guest) + gzippy-native (local arm64) sha 028bd002…cb410f T1+T8
path=ParallelSM. 856 lib tests pass (1 fail = pre-existing flaky diff_ratio timing micro-test, fails
identically on unmodified 507d6ecb). Seam + native_fold_parity green.

REMOVE-AND-MEASURE (NOT the SUM, advisor Q4): locked guest REDACTED_IP double-ssh, 16c gov=perf
turbo-on, taskset 0,2,4,6,8,10,12,14, T8, measure.sh interleaved N=11, RAW=68229982, sha-OK every run.
base(WITH merge) vs mergefix(REMOVED): run1 0.2291→0.2045 (+12.0%); run2 0.2128→0.1900 (+12.0%);
run3 0.2006→0.1765 (+13.7%, cleanest 6-13% spread). rg ratio base ~0.65× → mergefix ~0.73×. Sign
stable + load-invariant (1.64/2.80/1.86) ⇒ not a turbo artifact (interleaved = freq-neutral). KEEP.

ADVISOR: C1 UPHELD, C2 + C3 UPHELD-WITH-CAVEATS. Vendor citation accurate (change is MORE faithful);
no use-after-recycle on any emit path; re-resolution gates hold via !markers_resolved. Correction
ADOPTED: double-resolve tripwire (merge used to empty the buffer as the guard; safety now rests on
markers_resolved).

NEW WHOLE-SYSTEM WALL vs rg: T8 ~0.73× (was ~0.65×). Still a LOSS. NEXT (do NOT start — supervisor
gate): port (i) rg's multi-cached u16 marker loop (decodeBlock 1.69×, the larger gap), advisor-gated,
remove-and-measure. GUEST: /root/gzippy reset to clean 507d6ecb (prior overlays stashed as
owner-overlays-507turn) + /tmp/mergefix.patch. Builds /tmp/gzbuild-base (with merge) + /tmp/gzbuild-mergefix
(removed), both sha 028bd002…cb410f. Drivers /tmp/merge_measure.sh + /tmp/sha_check.sh (bash). NO orphans.

## CEILING BOUNDED → T8 TIE NEEDS TWO FAITHFUL PORTS (marker loop + view-based applyWindow); apply_window NOT at parity, divergence = a redundant memcpy [2026-06-07, OWNER turn, HEAD 507d6ecb +substep-timers-on-guest]
Paid the OWED apply_window measurement + source-verified rg's u16 marker-decode mechanism first-hand,
then DECOMPOSED gzippy's apply_window. Charter CURRENT STATE updated. Advisor:
plans/marker-kernel-ceiling-advisor-verdict.md (all UPHELD-WITH-CAVEATS, none refuted). Brief:
plans/marker-kernel-ceiling-brief.md. **SUPERVISOR GATE — fix build NOT started.**

RG MARKER MECHANISM (source, vendor deflate.hpp): readInternal (:1428) dispatches by Huffman-coding
TYPE not marker-vs-clean; with WITH_ISAL lit/len = readInternalCompressedMultiCached (:1453) for BOTH
u16 markers AND u8 clean (templated on Window). ONE loop; containsMarkerBytes = constexpr from element
type (:1600). Marker arms are cheap constexpr-gated only (dist-to-last-marker counter :1311-1317,
post-memcpy back-scan :1379-1389, inverse range-check skip :1652-1655); resolveBackreference fast arm
is std::memcpy for both (:1376). ⇒ NO separate slow marker path in rg; the 2× is gzippy's engine. The
faithful target is PORT rg's multi-cached u16 loop, NOT bolt AVX onto gzippy's loop (= E234 0.41×
plateau). Caveat: markers are u16 ⇒ ~2× clean traffic by construction; promise "marker == rg u16 loop."

OWED MEASUREMENT (locked guest REDACTED_IP double-ssh, 16c gov=perf turbo-on load~1.0, taskset
0,2,4,6,8,10,12,14, T8, RAW=68229982, sha 028bd002…cb410f every run, /tmp/gzbuild-isal native,
measurement-only sub-step timers byte-exact NOT committed, 3 runs):
  decodeBlock gzippy 0.838s vs rg 0.497s = 1.69×.
  apply_window decompose (SUM/15 marker chunks): gather 0.044-0.064s | crc 0.013-0.019s |
  MERGE 0.116-0.134s | subwin 0.010-0.012s | TOTAL 0.19-0.27s.
  rg --verbose first-hand: "applying the last window" = 0.032s (NOT charter's cached 0.113s = WRONG),
  checksum 0.0096s.
KEY: gather (rg's applyWindow analogue) is ~1.5-2× and ALGORITHMICALLY IDENTICAL (base[i]=lut[v] ↔
rg target[i]=fullWindow[chunk[i]], DecodedData.hpp:335-337). The DOMINANT divergence = MERGE
(chunk_data.rs:1589 → segmented_buffer.rs:356 prepend_narrowed_from_markers): a full ~68MB output-size
memcpy that rg does as std::swap + in-place VectorViews (DecodedData.hpp:368-388). gzippy already has
the zero-copy emit (append_narrowed_iovecs) ⇒ the merge-copy is REDUNDANT for the iovec writer.

ADVISOR: all UPHELD-WITH-CAVEATS, none refuted. Q3 merge removable byte-exactly + faithful (every
consumer supports un-merged state, traced) but a STRUCTURED change: defer marker-recycle behind
consumer writev, relax populate_subchunk_windows narrowed_len==0 assert (chunk_data.rs:1291), keep
narrowed_len through write. Q4 (LOAD-BEARING) do NOT trust −0.12s SUM as wall delta — merge runs on
the pool; wall cost = only the un-overlapped fraction on consumer recv_post_process_blocking
(chunk_fetcher.rs:1769) for un-pre-resolved head chunks, bounded by resolve-ahead hit rate
(project_confirmed_offset_prefetch_gap). Provable ONLY remove-and-measure. Q5 ceiling DIRECTIONALLY
SOUND, two ports right + faithful, NOT yet a proven TIE; third residual (gather/crc ~1.5× = segmented
walk + per-chunk LUT rebuild vs rg contiguous + hoisted fullWindow) under-counted.

BOUNDED CEILING (revised honest): T8 TIE plausibly pure-Rust via TWO faithful ports — (i) rg
multi-cached u16 marker loop (decodeBlock 1.69×) + (ii) rg view-based applyWindow = drop the redundant
merge memcpy (0.12-0.13s divergence). The prior "marker-COMPUTE only" ceiling was OPTIMISTIC exactly
as advisor Q4 warned. SCOPED FIX next loop (do NOT start): land MERGE-REMOVAL FIRST (cheapest, payoff
most uncertain ⇒ measure first) — swap+views model, defer recycle, relax assert, emit via
append_narrowed_iovecs; byte-exact + measure interleaved T8 wall (freq-neutral). THEN the multi-cached
u16 marker loop. Each advisor-gated, each remove-and-measure (never the SUM, never the slope).
GUEST: /root/gzippy @7bf26096 + oracle overlay + decompose knobs + THIS turn's measurement-only
sub-step timers in chunk_fetcher.rs (gather/crc/merge/subwin, via /tmp/patch_resolve.py +
/tmp/patch_merge.py, NOT committed locally, byte-exact). Build /tmp/gzbuild-isal (native, rebuilt).
Drivers /tmp/applywin_measure.sh + /tmp/substep2_measure.sh (bash). NO orphan processes.

## BUNDLE DECOMPOSED → T8 SUB-LEVER = marker-COMPUTE (gzippy window-absent u16 decode ~2× slower than rg) [2026-06-07, OWNER turn, HEAD 5e9905c8 +decompose-knobs]
Decomposed the GZIPPY_SEED_WINDOWS bundle (advisor's 3-removal confound) on the whole-system T8
wall. Charter CURRENT STATE updated. Advisor: plans/t8-decompose-advisor-verdict.md. Pre-reg:
plans/t8-decompose-prereg.md. Findings: plans/t8-decompose-findings.md. **SUPERVISOR GATE — fix
build NOT started (bound-ceiling-first; one owed measurement remains).**

WHAT: 2 measurement-only env knobs (OFF==identity, byte-exact, NOT committed): GZIPPY_SEED_NO_WINDOWS=1
(suppress seeded-window fallback ⇒ seed-only-boundaries) + GZIPPY_SEED_NO_BOUNDARIES=1 (skip
block_finder pre-seed ⇒ seed-only-windows). src/decompress/parallel/seed_windows.rs + chunk_fetcher.rs.

MEASURED (locked guest REDACTED_IP double-ssh, 16c gov=perf turbo-on load 1.3-2.0, measure.sh
interleaved N=11 CPUS=0,2,4,6,8,10,12,14 RAW=68229982 sha-OK=028bd002…cb410f every cell, 2 runs):
  rg 0.132s 1.000 | seedfull(both) 0.126-0.134s ~1.00× TIE | onlywin(windows) 0.199s 0.66× LOSS |
  onlybnd(boundaries) 0.198-0.205s 0.66× LOSS | prod(none) 0.198-0.203s 0.66× LOSS.
KEY: onlywin ≈ onlybnd ≈ prod (Δ<spread); ONLY seedfull (BOTH windows+boundaries) ties. f_windows≈0,
f_boundary≈0, yet seedfull ties ⇒ SUPER-ADDITIVE/COUPLED (pre-reg branch-4).
MECHANISM (GZIPPY_VERBOSE counters): seedfull window_seeded=17 spec-fail=0 Fill=91% decodeBlock=0.846s
(CLEAN). onlywin seed_hits=0 (windows UNUSABLE at partition-guess offsets) ≡ prod. onlybnd spec-fail
13→0 (real boundaries kill spec-failures) BUT body still 170MB/s u16 decodeBlock=1.106s ≈ prod ⇒
WALL-NEUTRAL. APPLES-TO-APPLES rg --verbose (both window-absent, same 34.5% markers): rg decodeBlock
0.542s vs gzippy prod 1.067s ⇒ rg's u16 marker decode ~2× FASTER per byte. rg ties WITHOUT seeding.

ADVISOR: core UPHELD-WITH-CAVEATS. Q1 the 2×2 knobs CANNOT separate marker-compute from boundary-
alignment — onlywin is DEGENERATE (windows unusable without boundaries by construction ⇒ ≡ prod,
pre-reg self-test FAILED ⇒ the COUPLED branch); re-attribute to onlybnd + rg-comparison. Q2 onlybnd
UPHELD-W-CAVEATS (spec-failures not the cost, wall-neutral). Q3 the 2× rate gap is FAIR (denominator-
matched, applyWindow separate in both, survives spec-failure removal) = STRONGEST pillar. Q4 (MOST
IMPORTANT) ceiling OPTIMISTIC: seedfull removes marker-premium AND applyWindow ⇒ bounds route-(ii)
not the faithful route-(i) (fast u16 marker decode KEEPS applyWindow); route-(i) ceiling rests on the
rapidgzip existence proof (rg 0.54 decode + ~0.113s applyWindow → 0.13 wall), conditional on
gzippy's applyWindow ≈ rg's.

PINPOINTED: T8 sub-lever = marker-COMPUTE (window-absent u16 decode ~2× rg). NOT boundary-alignment
(secondary precondition, wall-neutral) NOT spec-failures (wall-neutral). CEILING ≤ T8 1.0× TIE
CONDITIONAL on apply_window parity vs rg's ~0.113s.
NEXT (next loop, do NOT start now): igzip-class u16 marker-decode kernel (asm/inner-kernel techniques
adapted to u16 marker output — in scope HERE, Phase-0 ISA-L oracle never tested the marker path).
PLUS the OWED measurement: time gzippy's apply_window/marker-resolution vs rg's ~0.113s (needs a
fast-marker prototype or direct timer; no existing cell isolates it).
GUEST: /root/gzippy src @7bf26096 + oracle overlay + this turn's 2 decompose knobs (applied on guest,
NOT committed locally). Build /tmp/gzbuild-isal (gzippy-isal target-cpu=native byte-exact). Seeds
/tmp/seeds.bin (16 windows). Driver /tmp/decompose_measure.sh (use bash). NO orphan processes.

## PHASE-0 WALL ORACLE DONE → T8 BINDER IS THE WINDOW-ABSENT MARKER PATH, NOT THE ENGINE [2026-06-07, OWNER turn, HEAD 3895a23c +oracle]
PHASE-0 of the asm-port project: dropped a REAL ISA-L engine into the PRODUCTION parallel-SM
pipeline (pool/consumer/ring/CRC/window-publish kept) and measured the T8 WALL vs rapidgzip on the
locked guest. Charter CURRENT STATE updated. Advisor: plans/asmport-phase0-advisor-verdict.md.
Pre-reg + results: plans/asmport-phase0-prereg.md. Brief: plans/asmport-phase0-advisor-brief.md.
**SUPERVISOR GATE reached — Phase 1 (asm transliteration) NOT started, per the prompt.**

WHAT: `GZIPPY_ISAL_ENGINE_ORACLE=1` (measurement-only, env-gated, byte-exact, NOT production) routes
`finish_decode_chunk_impl`'s clean-tail decode through REAL ISA-L FFI
(`decompress_deflate_from_bit_with_boundaries`), feeding bytes/boundaries/end-bit through the SAME
ChunkData primitives. ISA-L input bounded to `[..stop_hint/8+256KiB]` (per-chunk, not whole-member —
the FIRST cut decoded the whole member per worker → 0.42s; bounding fixed it). Windows SEEDED
(`GZIPPY_SEED_WINDOWS`, captured T1) so all 18 chunks are window-present → reach the oracle. PROVEN
ISA-L ran: `isal_oracle_chunks=16 isal_oracle_fallbacks=1`.

MEASURED (locked guest REDACTED_IP double-ssh, 16c gov=perf turbo-on load 2.7-4.2, measure.sh
interleaved N=11 CPUS=0,2,4,6,8,10,12,14 RAW=68229982 sha-OK=028bd002…cb410f, 2 runs):
  rg 0.134s 1.000 | isal(ISA-L,seeded) 0.148s 0.905/0.892 TIE | pure(pure-Rust,seeded) 0.134s
  1.002/0.968 TIE | prod(pure-Rust,no-seed) 0.194s 0.690/0.652 LOSS.
KEY: `pure` (the SLOWER engine) ALREADY ties rg when seeded; engine swap is TIE-vs-TIE ⇒ the engine
is NOT the T8 binder. The ~1.5× prod gap collapses to TIE when the window-absent path is removed.
Per-stage: prod decodeBlock SUM 1.048s/Real 0.169s/Fill 77%/body 168MB/s/13 spec-fails; pure-seed
0.781s/0.108s/Fill 90.55%/0/0. rapidgzip runs the SAME 34.5% marker workload WITHOUT seeding yet
ties (rg --verbose) ⇒ gzippy's window-absent path is the slow one (apples-to-apples).

ADVISOR: C1 UPHELD-W-CAVEATS (oracle real, clean-tail only, 1-chunk impurity). C2 UPHELD-W-CAVEATS
(engine not the T8 binder; but 1.51× engine gap is REAL + slack-masked at Fill 90%, NOT at parity).
C3 UPHELD as COARSE localization — seeding is a valid causal removal + NOT unfair, BUT bundles 3
removals (u16 marker decode, block_finder real-boundary pre-seed vs partition GUESS, 13 spec-fail
re-decodes) ⇒ cannot isolate marker-COMPUTE from boundary-ALIGNMENT. C4 directional rec UPHELD at
T8, strong inference REFUTED (marker-phase rate is on the binding path + never replaced; T1 unaddressed).

NEXT (supervisor gate): DECOMPOSE the bundle before choosing Phase 1 — seed-ONLY-boundaries (no
windows) vs seed-ONLY-windows (prod boundaries). If the delta is boundary-ALIGNMENT, the lever is
the block finder / prefetch horizon (project_confirmed_offset_prefetch_gap), NOT the asm engine. If
marker-COMPUTE, a faster u16 marker kernel. The asm engine port is NOT the T8 lever (pure ties T8
seeded) but remains the plausible T1 lever + helps marker-phase rate — re-scope, don't abandon.
GUEST: /root/gzippy src @7bf2609 + this turn's overlay (oracle in gzip_chunk.rs + chunk_fetcher.rs);
builds /tmp/gzbuild-isal (gzippy-isal) + /tmp/gzbuild-native (gzippy-native), both byte-exact;
seeds /tmp/seeds.bin; driver /tmp/phase0_measure.sh (use `bash`, not sh — measure.sh needs bash).
Tree: oracle committed this turn; NO orphan processes (advisor + polls killed; guest clean).

## PURE-RUST ENGINE CEILING BOUNDED → PLATEAU (~0.6× ISA-L isolation); FORK real, WALL bound owed [2026-06-07, OWNER turn, HEAD f8260aa8 +bench]
Loop step (B) executed: built + measured the faithful-u8 engine CEILING vs ISA-L (isolation, bounded).
Charter CURRENT STATE updated. Advisor: plans/engine-ceiling-advisor-verdict.md (PLATEAU UPHELD-WITH-
CAVEATS). Brief: plans/engine-ceiling-advisor-brief.md. Bench committed this turn.

WHAT: added VAR_VI to benches/engine_isolation.rs = VAR_V speculative flat-u8 pipeline + igzip
packed-u32 table (tricks #1/#2/#3) PLUS (1) BMI2 BZHI distance-extra extraction, (2) AVX2/SSE MOVDQU
wide overlap-copy back-ref. trick #3 confirmed fully exploited.
MEASURED (locked guest REDACTED_IP double-ssh, 16c gov=perf load~3.3 turbo-on, taskset -c 0, N=11
interleaved, native target-cpu BMI2+AVX2 LIVE avx2_detected=true, 2 stable runs):
  ISA-L 847-851 MB/s | VAR_V 460-462 (0.54×) | VAR_VI 504-525 (0.59-0.62×); per-chunk VI 0.55-0.64.
  BMI2+AVX added ~9-14% over VAR_V, did NOT close the gap. SELFTEST PASS (iii/i=2.73).
BYTE-EXACT: VAR_VI printed MBps (never VOID) every chunk ⇒ byte-identical to scalar AND ISA-L over
the full timed window (bench gate exact[k]). Top-line SHA_ALL_EQUAL=no = PRE-EXISTING VAR_IV_E234
failures (untouched path), NOT VAR_VI.
FALSIFIER FIRED → PLATEAU: VAR_VI ≈0.6× ISA-L, ~23pp below the 0.85 PASS line, WITH the full igzip
stack + inline-ASM intrinsics. Pure-Rust igzip-class as a STANDALONE PRIMITIVE not reached.
ADVISOR: PLATEAU UPHELD-WITH-CAVEATS — all 5 techniques LIVE, fast loop is the timed path, header
symmetric, byte-exact airtight; crediting every caveat lifts a "fixed" VI to ≤~0.65-0.68, still
17-20pp short. LOAD-BEARING CAVEAT: escalating to "1.0× WALL hard-bounded at 0.6×" OVERREACHES
isolation (Rule 3 forbidden extrapolation); the WALL bound needs a PRODUCTION engine-removal oracle.
The floor-to-floor T8 1.74× finding (prior turn, advisor-upheld) independently corroborates the
engine gap reaches the wall, so the FORK is strongly implicated but the clean WALL hard-bound is owed.

NEXT (supervisor gate): (a) escalate the fork with the engine-primitive hard number (~0.6× ISA-L
PLATEAU) + corroborating wall evidence; OR (b, owner-recommended) run the production-pipeline
engine-removal oracle first to convert the engine ceiling into a clean WALL hard-bound, then escalate.
GUEST: /root/gzippy src @7bf26096 + this turn's bench overlay (benches/engine_isolation.rs has VAR_VI);
build /tmp/gzbuild (CARGO_TARGET_DIR), binary engine_isolation-9b415678a2bcb34c; /tmp/engine.seed ->
/dev/shm/engine.seed; corpus /root/gzippy/benchmark_data/silesia-gzip.tar.gz; /root/fulcrum stub present.
Tree: bench committed; no orphan processes (killed claude -p wrapper + sleep watchdogs).

## T8 BINDER RE-LOCATED to the ENGINE; "serial/consumer-wait" binder REFUTED as a unit error [2026-06-07, OWNER turn, HEAD f8260aa8]
Loop step (A) executed (perceive → ceiling-bound → causal-ID → advisor). Charter CURRENT STATE
updated. Advisor: plans/t8-engine-binder-advisor-verdict.md (claim1 UPHELD; 2,3 UPHELD-W-CAVEATS).
Brief: plans/t8-engine-binder-advisor-brief.md. NO code change this turn (perception + advisor).
Tree clean (only untracked plan/script files); NO orphan processes.

THE PRIOR BINDER ("decode floor 0.118s ≈ rapidgzip wall 0.130s ⇒ gap is all scheduling/consumer-
wait") IS A UNIT ERROR: it compared gzippy's FLOOR to rapidgzip's WALL. Floor-to-floor (both
--verbose Theoretical-Optimal): gzippy 0.118s vs rapidgzip 0.068s = 1.74×. decodeBlock SUM 0.93s
vs 0.50s = 1.86×. The T8 BINDER is the per-thread DECODE ENGINE (body_rate 269 vs ~424 MB/s).
future::get consumer-wait 0.077 vs 0.062 = 1.25× = MINORITY + downstream. Matches the constant
~1.7× at BOTH T1 and T8 (flat-across-T = per-thread throughput gap). DECODE-BYPASS + SLEEP-DECODE
oracles are CONFOUNDED (decode-free wall 3.6-5.5× SLOWER — buffer-pool bypass + per-chunk zeroed
allocs/faults + 660MB live + un-overlapped 212MB CRC); use floor-to-floor span comparison instead.
Vendor BlockFetcher.hpp:246-329: rapidgzip ALSO pumps prefetch in `while(wait_for(1ms))` during
future-wait — overlap structure already faithfully ported, no missing mechanism.

NEXT: step (B) = the ENGINE is the binder. Advisor caveat D-D (LOAD-BEARING): the round-2 "2.4×
plateau" that declared pure-Rust unreachable was on the DISCREDITED u16-RING arch — does NOT bound
the current faithful u8-direct engine (fc1c965b). pure-Rust→1.0× is OPEN. USER-CONSTRAINT FORK
implicated: resolve by BUILDING+measuring the faithful-u8 engine ceiling vs ISA-L on the prod
speculative path (igzip-class inner-Huffman: packed-u32 table, speculative 8B literal store +
next-sym preload, BMI2, MOVDQU overlap copy, slop headroom), NOT by extrapolating the u16 plateau.
GUEST: gzippy-mk2 byte-exact at /tmp; silesia /root/silesia.gz; cap_t1.bin+cap_t8.bin bypass
captures + measure_oracleA.sh/oracle_sleep.sh on guest /tmp (kept for reuse).

## [SUPERSEDED — unit error] T8 BINDER LOCATED = SERIAL/CONSUMER-WAIT; u16-path premise FALSIFIED [2026-06-07, OWNER turn, HEAD fb3baec0]
Loop step (A) executed in full (perceive → causal-ID → advisor → instrument committed).
Charter CURRENT STATE updated. Advisor: plans/u16-ceiling-advisor-verdict.md (UPHELD /
UPHELD-WITH-CAVEATS). Brief: plans/u16-ceiling-advisor-brief.md.

PREMISE CORRECTION (source + advisor UPHELD): the charter's "58.6% of bytes take the slow u16
path = biggest prize" was a MIS-READING of the mis-named counter BOOTSTRAP_POST_FLIP_U16_BYTES
(gzip_chunk.rs:97/:1302 increments on marker-FREE blocks, which since fc1c965b decode u8-DIRECT).
Genuine u16-<true> fraction ≈ 42.5% inverse (the pre-flip prefix), NOT the bulk-on-a-slow-path.

CAUSAL (new GZIPPY_SLOW_MARKER_MODE u16-path knob, commit fb3baec0, byte-exact 028bd002…cb410f
all knob modes; locked guest REDACTED_IP double-ssh, 16c gov=perf, measure.sh interleaved sha-OK,
RAW=211968000, T8 N=11; box load 3-5 ⇒ trust interleaved-relative only):
- CLEAN +100% spin → +27%; CLEAN +100% SLEEP → +27% (IDENTICAL = not turbo); +200% SLEEP → +55%.
  ⇒ clean u8 decode genuinely gates ~27% of T8 wall (freq-neutral).
- MARKER +200% spin → +21%; SLEEP control → +7% (collapses). ⇒ u16 marker = MINORITY ~3.5-14%.
- T1 MARKER → near-flat (validates: u16 barely runs at T1, knob fires ∝ u16 bytes).

BINDER LOCATED (pool trace, first-hand): decodeBlock 0.936s → Theoretical-Optimal(÷8)=0.117s;
Real Decode span 0.147s (Fill 79%); std::future::get in-order consumer wait = 0.077s; header_ms
24.0 (~2.6%). gzippy's perfectly-parallel decode FLOOR (0.117s) ALREADY ≈ rapidgzip's ENTIRE wall
(0.130s). The whole 1.70× gap = scheduling/serial: pool-fill gap ~0.030s + consumer head-of-line
future::get ~0.077s ≈ ~0.10s. rapidgzip ties DESPITE the same engine gap by overlapping decode.

NEXT: FIX the in-order consumer future::get head-of-line wait + pool-fill gap (charter binder #2 /
project_confirmed_offset_prefetch_gap). Bound the ceiling FIRST with a consumer-wait-removal oracle
(Rule 3). The prior placement-port gate FAILED on offset-supply (a non-divergence); the OPEN
distinct question is prefetch-HORIZON/dispatch-depth, not offset supply. Kernel stays the T1 lever
+ a real ~27% T8 term but is NOT the T8 path to 1.0×.

GUEST BUILD NOTE (next owner): guest src build needs a `/tmp/fulcrum` stub crate (Cargo.toml
optional dep `fulcrum = {path="../fulcrum"}` — cargo reads its manifest even with coz OFF) AND the
tarball must include crates/ + examples/ (workspace member + [[example]] coz_bench). Build dir
/tmp/gzbuild on guest; binary gzippy-mk2 (D2-fixed) byte-exact. silesia.gz is at /root/silesia.gz.

## T8 BINDER RE-IDENTIFIED — kernel REFUTED as the T8 lever; u16-path + scheduling bind [2026-06-07, OWNER turn] [u16 part SUPERSEDED above]
Full perceive→causal-ID→advisor loop. Charter CURRENT STATE updated. Advisor:
plans/t8-binder-advisor-verdict.md (UPHELD-WITH-CAVEATS). Brief: plans/t8-binder-advisor-brief.md.

WHOLE-SYSTEM WALL (locked guest REDACTED_IP -J neurotic, 16c gov=perf no_turbo=1, measure.sh
interleaved sha-verified, RAW=211968000): T8 gzippy-varv ~0.226s vs rapidgzip ~0.137s = 1.655×;
varv vs base TIE; sha 028bd002…cb410f OK. (gzippy-base 16:19 + gzippy-varv 16:18 on guest /tmp,
source /root/gzippy @ 7bf26096 + VAR_V worktree; rapidgzip 0.16.0.)

CAUSAL PERTURBATION (slow_knob, byte-transparent, freq-neutral sleep control, site fires ∝ clean
bytes; reproduced 3×, sha OK every run):
- T1 spin100 (doubles per-thread decode-compute) → +83% wall ⇒ decode GATES ~83% of T1 wall
  (kernel = confirmed T1 lever).
- T8 spin100 → +14–22%; spin200 → +45%; sleep100 control +20% (≥ spin, not a turbo artifact)
  ⇒ clean decode-compute gates only ~18–22% of T8 wall.
- COVERAGE CONFOUND (advisor Angle-4, reconciled first-hand): slow_knob is CLEAN-only (const-folds
  to 0 on marker <true>). clean hits T1=38.7M vs T8=28.4M (73% coverage) ⇒ ~27% of T8 events run
  MARKER mode uncovered. Coverage-corrected T8 decode ceiling ≈ 25–30% (advisor: ≤~45% w/ Rule-3
  unbind slack). Decode-compute is a MINORITY of the T8 wall regardless.

T8 BINDER (the ≥55–75% that is NOT clean decode-compute), GZIPPY_VERBOSE trace first-hand:
1. **u16 post-flip/marker path = 58.6% of decoded BODY bytes** (post_flip_u16_bytes=118.6M, trace
   labels it "Design-B1 prize"). Bulk of T≥2 bytes flow through the slow u16 marker→drain path,
   NOT the clean u8 fast path VAR_V/kernel optimized — why the clean gain was absorbed + slow_knob
   barely moved T8. Speculation header-failures 14/19; body_rate blended 286 MB/s.
2. **Pool sched + serial tail:** Theoretical-Optimal 0.127s → Real-Decode 0.162s (~28% pool
   inefficiency, fill 73–83%, dispatch saturated ~51/60) → wall ~0.22s (~0.06s SERIAL outside the
   pool: in-order publish / drain / CRC). Corroborates project_confirmed_offset_prefetch_gap.

RE-POINT (per PROCESS — bottleneck moved off the clean kernel at T8): NEXT = (A) ORACLE-bound the
u16-path ceiling (if 58.6% u16 body ran at clean-path rate, how much wall drops? likely the bigger
prize — rapidgzip ties DESPITE the same engine gap via in-flight overlap); (B) pool-sched/serial-
tail SERIAL-WORK-vs-DECODE-WAIT decomposition. Inner kernel KEPT as confirmed T1 lever, NOT the T8
path to 1.0×. No build this turn (perception + causal ID + advisor only). Tree clean; no orphans.
NOTE for next owner: guest reachable ONLY as `ssh neurotic 'ssh REDACTED_IP "…"'` (local key not
authorized on guest; -J neurotic from local FAILS the final hop). Watch leftover `timeout` sleep
watchdogs after claude -p / measurements — kill -P the wrapper pid.

## WINDOW-DISCARD LEVER — SOURCE-VERIFIED + QUANTIFIED → **FALSIFIED** [2026-06-07, leader fresh instance]
Charter: plans/window-discard-and-overhead-diagnosis.md. The advisor flagged
`reset(None, window_opt)` (gzip_chunk.rs:1107) as discarding an available predecessor window.
**FULL CALL-GRAPH TRACE REFUTES IT.** No window is discarded.

STEP 0: VAR_V committed as 9b674651 (byte-exact TIE kept per rule 7a, gain latent).

WHY NO DISCARD (source, first-hand):
- gzip_chunk.rs:1107 `block.reset(None, window_opt)` — `output=None` makes the seed guard
  (`marker_inflate.rs:590 if let (Some(out),Some(window))`) false. BUT its ONLY caller,
  `decode_chunk_unified_marker` (gzip_chunk.rs:743), hard-codes `marker_decode_step(..., &[], ...)`.
  So `initial_window=&[]` → `window_opt=None` ALWAYS at :1107. It discards nothing — there is no
  window on that path to begin with. (The seed `output` param is only a precondition gate; the
  seed actually writes the predecessor into `output_ring`, marker_inflate.rs:709 — not into output.)
- The window-PRESENT case is handled SEPARATELY + EARLIER: `decode_chunk_with_rapidgzip_impl`
  (gzip_chunk.rs:598) `if initial_window.len()==MAX_WINDOW_SIZE` → finish_decode_chunk_impl /
  _with_inexact_offset WITH the window = the clean seeded fast path (vendor setInitialWindow). It
  NEVER reaches the marker bootstrap. Worker dispatch (chunk_fetcher.rs:2323) routes a
  `window_map.get(start_bit)==Some` chunk straight there with materialized window bytes.
- The +8.7% window-seed (commit 9949d2f0) is LIVE for window-present chunks via this :598 path;
  it was implemented in MarkerRing (now `GZIPPY_MARKER_RING=1` legacy only) but the production
  vendor-Block path has its OWN equivalent seeded route at :598. Not regressed.

QUANTIFIED (silesia 68MB, pure-rust-inflate, GZIPPY_VERBOSE, path=ParallelSM, ~18 chunks):
- T1: window_seeded=16, finished_no_flip=0  → EVERY non-zero chunk seeds its window.
- T2/T4: window_seeded=2, finished_no_flip=15 → ~89% window-absent.
- T8: window_seeded=2, finished_no_flip=16 → ~89% window-absent.
The T1→T8 collapse is the CAUSE: at T1 the in-order consumer publishes each window before the
single worker needs the next chunk; at T≥2 workers race AHEAD of in-order publication →
`window_map.get` returns None → speculative marker bootstrap. This is the FAITHFUL vendor model
(rapidgzip is also ~97% window-absent at runtime, chunk_fetcher.rs:2235-2250 non-blocking get).
The 16 absent chunks at T8 had NO predecessor window AVAILABLE-AND-DISCARDED — they were
genuinely not-yet-published. **No discard. No lever here.**

FALSIFIER (pre-registered, then RESOLVED): "If a window were discarded, T1 and T8 would BOTH
show high finished_no_flip (the seed never fires regardless of availability)." MEASURED: T1
shows finished_no_flip=0 / window_seeded=16. The seed DOES fire whenever a window is available.
Falsifier triggered ⇒ window-discard claim REJECTED.

NEXT (per charter step 2, NOT yet started — gated): the real binder is production clean-path
OVERHEADS that absorbed VAR_V's +48% isolation gain (ring %U8_RING_SIZE / reversed-bits dist /
backref arms / resumable cap). The only window-related angle that remains FAITHFUL is closing the
worker-vs-consumer publication race so MORE chunks seed at high T — but rapidgzip has the same race,
so that is parity-matching not a gzippy-only fix. STOPPED at supervisor gate before any large build.



## VAR_V INTEGRATION + REAL-WALL MEASUREMENT — DONE [2026-06-07, leader fresh instance]
Charter: plans/varv-integrate-and-measure.md. VERDICT: byte-exact, WALL TIE, decode STILL binds.

### INTEGRATION (faithful, real overheads kept — NO shortcut)
- VAR_V speculative software-pipelined fast loop ported INTO production
  `read_internal_compressed_specialized::<false>` (marker_inflate.rs ~:1454), the clean
  post-flip path = 89% of production (trace: finished_no_flip=16/18 chunks). Ported ONTO the
  REAL production wrapping u8 ring (`% U8_RING_SIZE`) — NOT a flat buffer. Kept ALL the overheads
  the bench elided: ring modulo + wrap-straddle (back-ref via the EXISTING `emit_backref_ring_u8`),
  the resumable `n_max_to_decode` cap (top-of-loop guard `emitted+FAST_OUT_SLOP < n_max_to_decode`),
  drain, CRC (downstream sink unchanged). Distance via production's `self.dist_hc` (cached
  reversed-bits), NOT the bench's LutDistCode — faithful + byte-exact. Fast loop runs only while
  the physical dst region won't wrap (`dst_phys+FAST_OUT_SLOP<=U8_RING_SIZE`) + input slop; falls
  to the unchanged careful per-symbol loop near wrap / resumable boundary / block tail. Const-folds
  away entirely on the marker `<true>` path. (No commit yet — STOPPED at supervisor gate.)
- BYTE-EXACT (dual-feature, dual-arch): gzippy-native arm64 AND gzippy-isal x86_64 (Rosetta
  cross-compile) BOTH emit silesia sha 028bd002...cb410f via path=ParallelSM, T1 AND T8. Guest
  (x86_64) native build also sha-exact. Tests: routing 37/0, pure_rust_inflate_corpus, the
  adversarial seam test (faithful_u8_flip_seam_max_distance_backref_vs_flate2) + native_fold_parity
  all GREEN (--features gzippy-native, --test-threads=1).

### MEASUREMENT (leader-run, locked guest trainer=REDACTED_IP -J neurotic, 16c, gov=performance)
- Built base (fa9fd73c, VAR_V reverted via patch -R) + varv (HEAD+VAR_V) on guest,
  RUSTFLAGS=-C target-cpu=native, both sha-exact. rapidgzip 0.16.0 same sha. RAW=211968000.
  measure.sh interleaved N=11, taskset-pinned, output-sha-verified=OK every run.
- T8 (CPUS=0,2,4,6,8,10,12,14): run1 varv 0.2145s vs base 0.2220s = 1.035x TIE (spread 10%);
  run2 varv 0.2213s vs base 0.2137s = 0.966x TIE (spread 31%). Sign FLIPS between runs ⇒
  noise-dominated TIE, mean ≈ 1.00x. rapidgzip 1.62–1.71x vs base (gap UNCHANGED).
- T1 (CPUS=0): varv 0.5165s vs base 0.5256s = 1.018x TIE (spread 7%). rapidgzip 1.702x vs base
  (gap UNCHANGED — identical to the pre-VAR_V faithful-u8 measurement's 1.702/1.704x).
- PER-STAGE TRACE (GZIPPY_VERBOSE, T8): decodeBlock base 1.227s vs varv 1.248s (= same, +1.7%
  noise); Pool Fill Factor 82.25% (base) / 82.65% (varv) — workers ~83% busy DECODING, placement
  overhead only ~17%. finished_no_flip=16 (VAR_V IS on the production bulk path).

### VERDICT — decode STILL binds; VAR_V's 0.555 did NOT survive integration
- The wall is a TIE at BOTH T1 and T8; the rapidgzip gap is CONSTANT ~1.70x (unchanged from the
  prior faithful-u8 TIE). The §3 projection (decode_wall 0.410s, "decode stops binding") is
  FALSIFIED on the real wall: producer-side `decodeBlock` is UNCHANGED (1.227→1.248s) — the
  speculative-store gain measured in the flat-buffer bench (+48% vs scalar, 0.555x ISA-L) was
  ABSORBED by the real ring/wrap/resumable overheads the bench elided (exactly the advisor's
  "0.555 is optimistic, real integration ≤ 0.555" caution, now confirmed quantitatively).
- CONTINGENT NEXT LEVER (per charter): decode STILL binds ⇒ the inline-asm/BMI2 spike to push the
  rate is the live direction (NOT placement — placement is ~17% slack here, fill-factor 83%).
  Do NOT start it this turn. The byte-exact VAR_V integration is a TIE — KEEP per rule 7a
  (correct + faithful inner-loop port) but it is NOT itself the wall lever.
- Independent disproof advisor: plans/varv-integration-advisor-verdict.md (synchronous, read-only).
- STOPPED for supervisor gate.

## FAITHFUL u8 NATIVE REWRITE — IN PROGRESS [2026-06-07, leader fresh instance]
Charter: plans/faithful-u8-native-rewrite.md. Leader-lock held (pid in /tmp).
### STEP 1 DONE — production clean path ESTABLISHED (source-map subagent, exit 0, all file:line first-hand)
- TWO native clean-bulk paths exist, selected by whether a 32KiB predecessor window is known:
  (A) window-SEEDED chunks -> ResumableInflate2 via StreamingInflateWrapper (resumable.rs:469/494/507
      writes u8-DIRECT into caller's &mut [u8]) -> ALREADY FAITHFUL u8, no u16, no narrow.
  (B) window-ABSENT/speculative chunks -> Engine M marker_inflate::Block (output_ring: Box<[u16;65536]>,
      marker_inflate.rs:290) -> the u16 ring + post-flip narrow-at-drain (drain_to_output :738-750).
      THIS is the only u16 storage left; the FOLD (gzip_chunk.rs:1205) keeps Engine M running the
      clean tail in-place on native (FlipToClean gated behind isal_clean_tail = OFF).
- PRODUCTION MEASUREMENT (silesia T8, gzippy-native arm64, GZIPPY_VERBOSE, sha 028bd002...410f VERIFIED):
  Unified decoder: finished_no_flip=16  flip_to_clean=0  window_seeded=2  inflate_wrapper=0  finish_decode=2
  => 16/18 chunks (89%) go through Engine M's u16 ring (finish in marker+clean mode, NO flip), only 2
  window-seeded (u8). So the u16 ring IS the production bulk path; the engine-bench PLATEAU measured a
  representative architecture. The faithful u8 rewrite of Engine M is on-target (NOT a dead path).
- The prior u8-direct port (5256075) landed in lut_bulk_inflate.rs::MarkerRing (conflate_to_clean_u8),
  reached ONLY via GZIPPY_MARKER_RING env (gzip_chunk.rs:1018) — DEAD on production. NOT on Block.
- Pre-req build fix: bench-only E2-E4 (read_clean_e234 / emit_backref_ring_clean / emit_avx2_copy_u16,
  7bf26096) used x86_64-only AVX2 intrinsics under a pure_inflate_decode cfg that also fires on arm64 =>
  arm64 build broke. Gated those three behind all(pure_inflate_decode, target_arch="x86_64"). Byte-
  transparent (dead bench code; production sha unchanged 028bd002...410f).
### STEP 2 — u8-width flip-in-place rewrite of Engine M: DONE (committed fc1c965b + test hardening).
- The faithful u8-direct rewrite LANDED byte-exact. Engine M post-flip now decodes the clean bulk
  u8-DIRECT into the u8 VIEW of the SAME output_ring backing (vendor getWindow() reinterpret
  deflate.hpp:890-894); flip = drain_transition_narrow_u16 (one-shot) + flip_repack_to_u8 (vendor
  setInitialWindow conflation 1762-1782: value-DOWNCAST (x&0xFF) the rotated 32KiB window into the u8
  view upper half, re-base cursor to u8-logical BASE=U8_RING_SIZE). emit_backref_ring_u8 = faithful u8
  port (8-byte word copy, distance>=8, RLE memset, overlap), no marker scan. set_initial_window_impl +
  read_internal_uncompressed store u8 in clean mode. Marker (<true>) path const-folded byte-IDENTICAL.
- BYTE-EXACT: silesia T1/T8/T16 == 028bd002...410f on gzippy-native (arm64) AND gzippy-isal (x86_64
  Rosetta cross-compile); 851 lib tests; clippy clean. MANDATORY adversarial seam test added +
  HARDENED: faithful_u8_flip_seam_max_distance_backref_vs_flate2 (6×A=192KiB so flip fires ~65278 and
  the distance-32768 back-refs are UNAMBIGUOUSLY post-flip; asserted vs independent flate2 oracle;
  sentinel at byte 163840 proves the repack rotation+downcast). native_fold_parity + the prior seam
  unit tests all green.

### STEP 4 — MEASUREMENT (leader-run, locked guest REDACTED_IP via -J neurotic): VERDICT = byte-exact TIE.
- Guest: 16 cores, governor=performance, load ~1.0 (~6% on 16c, measure.sh did NOT flag busy). no_turbo
  write was Permission-denied (guest VM; host controls turbo) — measure.sh's INTERLEAVED RELATIVE delta
  is turbo-immune (both tools see the same per-trial state), so the ratio is the authoritative signal.
- Uploaded /tmp/silesia.gz (decoded sha 028bd002...410f VERIFIED on guest). Built BOTH binaries on guest
  (RUSTFLAGS=-C target-cpu=native): /tmp/gzippy-base (u16 baseline from git HEAD 7bf2609) + /tmp/gzippy-u8
  (synced u8 source). Both byte-exact via path=ParallelSM.
- 3-WAY INTERLEAVED measure.sh (RAW=211968000, sha-verified=OK, taskset-pinned):
    T8 (CPUS=0,2,4,6,8,10,12,14, N=11): base 0.219s/968MB/s (spread 31%); u8 0.218s/972MB/s (spread 14%);
      rapidgzip 0.1285s/1650MB/s.  => u8 1.004x vs base = TIE; rapidgzip 1.704x vs base.
    T1 (CPUS=0, N=11 then N=15): base 0.521-0.526s/403-407MB/s (spread 7-9%); u8 0.537-0.539s/393-394MB/s
      (spread 4%); rapidgzip 0.306s/694MB/s.  => u8 0.976x vs base = TIE; rapidgzip 1.705x vs base.
- FINDINGS:
  1. The faithful u8 rewrite is a byte-exact TIE at BOTH T1 and T8 (Δ 2.4% << spread). KEEP (faithful,
     correct, layer-don't-revert).
  2. The gzippy->rapidgzip gap is CONSTANT ~1.70x at T1 AND T8. A flat ratio across T is the signature of
     a PER-THREAD decode-throughput gap (decode BINDS), NOT scheduling/placement.
  3. CAUSAL DISPROOF of the traffic hypothesis: the u8 rewrite is a clean ~2x perturbation of clean-path
     memory traffic (u16->u8 ring + u8 copies + no narrow) with a FLAT wall response => traffic is SLACK;
     the binding term is per-symbol LUT-DECODE COMPUTE. This FALSIFIES round-2's "u8 clean ring is the
     main lever" and confirms the round-2 PLATEAU (engine ~2.4x ISA-L on compute). The advisor's prior
     prediction "flat-u8 stops binding above ~120MB/s" is FALSIFIED: u8 = 393MB/s single-core, still
     1.70x off rapidgzip, decode still binds.
  4. CAVEAT (not load-bearing): measure.sh absolutes (393-972 MB/s) are a different sink (stdout->mktemp)
     than the charter's "0.604s same-sink wall" — NOT apples-to-apples. The verdict is RELATIVE (u8 vs
     base, same harness) and stands; the 0.604s absolute question is answered only as "u8 did not change
     the relative gap to rapidgzip."
- INDEPENDENT DISPROOF ADVISOR (synchronous, read-only, plans/faithful-u8-native-advisor-verdict.md):
  A byte-exact UPHELD; B seam UPHELD-WITH-CAVEATS (sentinel pinpoint softer than advertised — FIXED by
  the test hardening above); C wall-TIE UPHELD (TIE band sound; mechanism inference is a valid causal
  disproof; cross-harness 0.604s not load-bearing). Mandate SATISFIED structurally. REAL next lever =
  inner Huffman LUT-decode compute kernel (igzip-class packed-table/speculative-store/preload), NOT
  traffic/placement/ring.
- STOPPED for supervisor gate. The asm-kernel port + gzippy-isal u8+FFI are LATER/contingent.

## ASM-KERNEL FEASIBILITY SCOPE — CHECKPOINT REACHED [2026-06-07, leader fresh instance]
Charter: plans/asm-kernel-feasibility-scope.md. ANALYSIS/DESIGN ONLY, no build. Answer the
three questions (igzip-fast map / faithful integration cost+fork / tie projection), write
plans/asm-kernel-feasibility-report.md, route a synchronous disproof advisor, then STOP.
- igzip kernel source map DONE (synchronous read-only subagent, all file:line first-hand).
  KEY: the 4 load-bearing fast tricks (packed-u32 short table, speculative 8-byte literal store
  + next-sym/next-dist preload pipeline, MOVDQU overlap-doubling copy, slop-margin headroom
  guard licensing unchecked over-read/write) ALL DEPEND on the flat-u8 output+window-in-place
  model (asm:518/591/605/618, C:1641/1698). igzip uses SSE-width MOVDQU (xmm), NOT AVX2 ymm —
  the "AVX2"/Haswell build differs mainly in BMI2 (SHLX/SHRX/BZHI), not vector width.
- DELIVERABLE: plans/asm-kernel-feasibility-report.md + plans/asm-kernel-feasibility-advisor-verdict.md.
- STOPPED for supervisor + USER ratification (faithfulness fork is a user-level call). No build.

## ENGINE BENCH ROUND 2 — IN PROGRESS [2026-06-07, leader fresh instance]
Charter: plans/engine-bench-round2-authorization.md. SETTLE the plateau falsifier by building
E2-E4 in the engine-isolation bench (standalone, NOT production integration).
- STEP 0 DONE: round-1 harness COMMITTED 5d5fc3b9 (pushed). Self-test band RECALIBRATED
  [1.7,2.6]->[2.5,3.6] (round-1 FAIL was mis-calibrated: pure-ISA-L is a purer denominator
  → larger honest ~3.1x ratio, advisor-confirmed iii/ii=3.10x).
- KEY STRUCTURAL FACT (source-verified): production clean dynamic-block loop is
  read_internal_compressed_specialized<false> (marker_inflate.rs:1191) — the multi-cached LUT
  loop. Back-ref copy = emit_backref_ring<false> (:2137), writes into a u16 RING. Literal store
  = single u16 (:1348). Refill = bits.refill() once per outer iter (:1324). THESE are the E2/E3/E4
  targets. The u16 ring means every byte costs 2 bytes of mem traffic vs ISA-L's u8 buffer — the
  prime suspect for the 3.1x gap (compute is already heavily optimized: word-copy, RLE fill, etc).
- E2 (wide SIMD back-ref copy): emit_backref_ring already does 8-byte-word(=4 u16) copies + RLE
  fill. Headroom = wider vectors (AVX2 32-byte = 16 u16) for the longer matches; main lever is
  CLEAN u8 ring (halves copy traffic). E3 (packed multi-literal store): :1343 unpacks the
  TRIPLE_SYM 1-3 packed literals one u16 at a time — collapse to one wide store. E4 (wide refill):
  refill is already libdeflate-style (56-63 bits) once/iter; headroom = amortize over >1 symbol.
- PLAN: prototype E2-E4 as new bench variants (byte-exact gate vs scalar + ISA-L), measure
  separately + stacked, chunk-sweep 3-5 clean chunks, settle PASS/PLATEAU via tier1-design §3.
  Delegated to a synchronous subagent (spec in plans/engine-bench-round2-impl-spec.md).

### ROUND-2 SETTLED — VERDICT: **PLATEAU** [2026-06-07, committed 7bf26096]
- COMMITTED 7bf26096 (pushed): bench-only E2-E4 in marker_inflate.rs (read_clean_e234<E2,E3,E4>
  + drain_clean_u8 + emit_backref_ring_clean + emit_avx2_copy_u16, all cfg-gated like
  read_internal_compressed_specialized; production read() dispatches ONLY to <true>/<false> —
  read_clean_e234 has ZERO production callers, grep-verified). Added VAR_IV_E000 (<false,false,false>)
  byte-exactness anchor to the bench.
- BYTE-EXACT GATE: GUEST run SHA_ALL_EQUAL=yes on all 5 swept clean silesia chunks for all 7
  variants incl VAR_IV_E000 (read_clean_e234<false,false,false> == scalar <false> == ISA-L,
  byte-identical) AND the E2/E3/E4 stacks (AVX2 LIVE on guest, avx2_detected=true). Also SHA-clean
  under Rosetta (scalar fallback). Guest freq-locked: no_turbo=1, GATE PASS, RESTORE VERIFIED.
- GUEST AGGREGATE (median-of-per-chunk-medians, MB/s, 5 chunks, taskset core0, N=11 interleaved):
    VAR_I scalar      104   (=92.7ms/chunk anchor; projects 0.615s == design's 0.6134s — model self-consistent)
    VAR_IV_E000       108   (engine, no technique; +4% over scalar)
    VAR_IV_E2         121 / VAR_IV_E23 120 / VAR_IV_E234 (stacked) 118
    VAR_III ISA-L     283
- **(ii_stacked E234)/(iii ISA-L) = 0.412** (per-chunk 0.356..0.466, sd 0.036). This is FIRMLY in
  PLATEAU: ≤ the pre-registered 0.65 plateau line, **12.2σ below** the 0.85 PASS threshold. E234
  adds only ~8.7% median over E000 — the engine stays ~2.4× SLOWER than ISA-L (pure-decoder class).
- §3 PROJECTION (anchored 92.7ms↔104MB/s, 39 chunks, ramp 1.36, T8): E234 projects decode_wall
  0.542s. APPEARS ≤ the 0.604s tie bar, BUT this is a NUMERICAL ARTIFACT — decode_wall(0.542)≈
  total_wall(0.542), i.e. DECODE IS STILL THE BINDING TERM (it did NOT re-bind onto the ~0.54s
  shared floor). The bar sits right at the scalar floor (0.615s) so a ~13% engine bump trips it
  without decode ceasing to bind. The pre-registered PASS requires (ii)/(iii)≥0.85 so decode stops
  binding; E234 (0.41) is ~2× short. RATIO criterion GOVERNS → PLATEAU.
- FALSIFIER FIRED: residual gap to igzip-class (0.85-0.41=0.44) >> spread (sd 0.036, 12σ). Per the
  pre-registered falsifier: engine front is **NOT PROVEN** in pure-Rust+ASM as prototyped. The
  1.0× bar is NOT reachable via this E1-E4 inner-loop direction without FFI (or a revisited bar).
  SUPERVISOR/USER-LEVEL FINDING. Do NOT integrate; do NOT start the multi-week engine build.
- WHY pure-Rust+ASM plateaus (mechanism, not just a TIE): vendor's own bench has ISA-L at 2.1× the
  best PURE decoder single-thread; this bench reproduces ~2.4×. The gap is the u16 ring (2 bytes/byte
  mem traffic vs ISA-L's u8 buffer) + igzip's whole-AVX2 codegen; E2 (AVX2 copy on the STILL-u16 ring)
  + E3 (packed store) + E4 (amortized refill) recover only ~9% because the dominant cost is the per-
  symbol LUT-decode + u16 traffic that these techniques don't remove. A true E1 (u8 ring) was NOT
  prototyped here (faithful u16 ring kept for byte-exactness); it is the remaining untested lever but
  would diverge the ONE-engine ring storage from vendor's m_window16 — a supervisor call.
- NEXT (SUPERVISOR GATE): independent disproof advisor verdict ->
  plans/engine-bench-round2-advisor-verdict.md, then STOP for supervisor. No production integration.

## PLACEMENT PORT GATE — **FAILED (do NOT code attempt #4)** [2026-06-07, HEAD e52b0fc2]
Charter: plans/placement-port-authorization.md. The HARD 3-prior-failures re-derivation
gate ran (2 read-only subagents + 1 independent disproof advisor, all synchronous). Verdict:
**GATE FAIL — STOP. No port code written.** Full advisor verdict: plans/placement-port-advisor-verdict.md.

- The authorized lever's PREMISE IS FACTUALLY WRONG. Authorization said "gzippy never
  re-targets an overshot index at its CONFIRMED offset; rapidgzip does (GzipBlockFinder.hpp
  :117-158)." Source (leader + advisor + leader-reverified) shows gzippy ALREADY re-targets
  at the confirmed offset at THREE faithful sites: gzip_block_finder.rs:180-182 (confirmed
  idx→confirmed offset), chunk_fetcher.rs:1306 (matches_encoded_offset accept) + :1431
  (get_with_prefetch cold get AT the confirmed offset), block_fetcher.rs:945 (submit_for uses
  block_finder.get(index) = confirmed offset). The lever targets a NON-DIVERGENCE.
- Leader-VERIFIED directly (not just advisor): `needs_confirmed_offset` has ZERO hits in src/;
  block_fetcher.rs:784-790 pushes BOTH the confirmed off AND the partition offset (partition
  secondary, exactly vendor BlockFetcher.hpp:485-490) — gzippy does NOT collapse to partition.
- VENDOR reaches the SAME overshoot cold-get (GzipChunkFetcher.hpp:646-654 fallthrough to
  on-demand decode at the real offset). It absorbs it via the 2·P guess-prefetch horizon +
  pump-during-wait, NOT a distinct block-finder mechanism. The confirmed offset is born at the
  in-order insert in BOTH (appendSubchunksToIndexes :357-375 ↔ consumer_append_subchunks_vendor
  chunk_fetcher.rs:2788-2795) = ≤1-chunk lead in both. No lead-lengthening mechanism to port.
- WHY prior 3 failed (corroborated): all supplied confirmed offset at the ≤1-chunk frontier;
  Attempt 3 submitted the EXACT correct offset → fetcher_get UNCHANGED T8 449ms / WORSE T16
  303→936ms (in-flight-not-done). The proposed port inherits the SAME ≤1-chunk lead AND has
  nothing to change → attempt #4 with the same load-bearing constraint.
- ANTI-ESCAPE-HATCH (advisor guardrail, NOT yet done): do NOT bundle this into "it's the
  engine, placement done." The genuinely-distinct UNCLOSED structural question: stalls are
  "all decode_NOT_STARTED" — either workers saturated (engine) OR the GUESS-prefetch for that
  index was never dispatched DEEP ENOUGH AHEAD (prefetch-HORIZON/scheduling — structural,
  distinct from offset supply). That is the pre-registered slow-knob perturbation in
  placement-rescope-diagnosis.md. **Proposed next step (SUPERVISOR GATE): run the
  decode_NOT_STARTED / prefetch-horizon perturbation — NOT the offset-supply port, NOT the
  engine build.** STOPPED for supervisor.


## NEW MISSION (2026-06-06, user-set) — SUPERSEDES all prior arcs
Prior "full ISA-L port as a 2nd engine" plan is DEAD (critically reviewed: redundant
2nd inflate vs ONE-engine memory; x86-only strands arm64). This mission, in order:

- PHASE 1 — NAMING TRUTH PASS (behavior-neutral renames). Every function/type/module/
  dataset/span whose name misdescribes behavior gets renamed; all call sites, doc
  comments, trace/span labels updated. FLAGSHIP: IsalInflateWrapper on the pure-Rust
  build does NOT wrap ISA-L — it delegates to pure-Rust Inflate. Audit whole
  src/decompress/parallel/ + src/decompress/inflate/. Gate: build + byte-exact tests +
  silesia sha 028bd002c89c9a909ccdbc2af0a223de285348edb014ccc8e27d297f52cb410f.
  Commit: "rename(parallel/sm): naming truth — names describe behavior".
- PHASE 2 — TWO FLAG-GATED PATHS + DELETE dead.
  - gzippy-isal feature: FAITHFUL rapidgzip — marker engine + 32KiB clean-handoff to
    REAL ISA-L via FFI (vendor GzipChunk.hpp:520-525). C-FFI behind THIS flag ONLY.
  - gzippy-native feature (DEFAULT prod): UNIFIED ONE-ENGINE marker decode, single pass,
    pure Rust, flip-in-place clean tail, NO 2nd engine, NO FFI, ISA-L hot techniques
    grafted (BMI2 PEXT/BZHI dispatch OFF unified.rs:122; multi-symbol LUT; lean refill).
  - Each path BYTE-EXACT. native also diff-tested vs gzippy-isal oracle + flate2.
    DELETE flagged-off unused impls. Advisor-gate each deletion.
- PHASE 3 — FULCRUM 3-WAY: locked harness (neurotic, T8 silesia, interleaved best-of-N,
  sha-verified) for rapidgzip vs gzippy-isal vs gzippy-native. Report 3-way wall + d_c/d_w.

## CONSTRAINTS
OOM: serialize ALL builds (pgrep first; ONE globally; WAIT). Prefer neurotic. Never two
building subagents. No duplicate orchestrators. Long ops detached + sentinel, polled.
Byte-exact ABSOLUTE. Delegate ALL hands-on work; advisor-corroborate consequential claims.
Numbers ONLY from full locked harness. Clean base HEAD 412b3ac. KEEP #B + byte-exact.

## PHASE 1 — IN PROGRESS
- Worktree: .claude/worktrees/phase1-naming-truth (branch phase1-naming-truth from 412b3ac).
- Observed: inflate_wrapper.rs has TWO IsalInflateWrapper structs (line 154 real-ISA-L cfg,
  line 340 pure-Rust cfg). 9 src files reference IsalInflateWrapper.
- STEP 1 [running]: design subagent produces rename MAP (old->new + rationale). Advisor-
  review. Then impl subagent applies. Then gate.

### STEP 1 DONE — rename MAP produced + advisor-reviewed (APPROVE WITH CHANGES)
- Governing fact (advisor-confirmed via build.rs:81-83): pure_inflate_decode==parallel_sm;
  the old ISA-L decode arm was REMOVED; NO live real-ISA-L decode in parallel/+inflate/.
  Every isal-named symbol there is FALSE-isal (pure-Rust). TRUE-isal lives only in
  src/backends/ + `feature=isal-compression` GATES (not names) — DO NOT rename.
- Approved renames: IsalInflateWrapper->StreamingInflateWrapper (both arms, ~28 code +
  exclude ~12 vendor doc refs); isal_huffman_pure.rs->lut_huffman.rs (mod.rs:46);
  IsalLitLenCodePure->LutLitLenCode (10); IsalDistCodePure->LutDistCode (6);
  isal_lut_bulk.rs->lut_bulk_inflate.rs (mod.rs:48, 19 path refs);
  isal_litlen_pure->lut_litlen (4); isal_lut_litlen_rebuild->lut_litlen_rebuild (3);
  isal_lut_litlen_decode->lut_litlen_decode (2).
- DEFER: trace span worker.isal_stream_inflate (cross-coupled to patch_vendor.sh; renaming
  one side breaks `fulcrum vs` — not behavior-neutral for tooling). KEEP ISAL_* vendor
  constants (traceability). 
- ADD (advisor): also rename DecodePath::IsalParallelSM/IsalSingle in src/decompress/mod.rs
  (flagship FALSE-isal; the GZIPPY_DEBUG path= contract) -> ParallelSM/SingleMember.
  Update GZIPPY_DEBUG label + routing tests + CLAUDE.md mentions consistently.
- STEP 2 [DONE 2026-06-06]: rename applied + gated + COMMITTED.
  - Commit 6b7d6f5 on branch phase1-naming-truth (worktree .claude/worktrees/phase1-naming-truth),
    parent 412b3ac. Diff symmetric pure-rename + a cargo-fmt pass (hook-required: import
    reorder/line-wrap of the renamed code; still behavior-neutral).
  - Gate (deadlock-safe — SINGLE serialized cargo runs, timeout-wrapped, --test-threads=1,
    load-flaky tests skipped):
      * build clean (release, --features pure-rust-inflate); cargo fmt --check clean.
      * lib tests 35 passed / 0 failed (routing:: + pure_rust_inflate_corpus::; skipped
        not_slower/diff_ratio/scoped_cancel/hot_path/alloc_budget).
      * silesia sha256 028bd002...410f == reference, emitted via path=ParallelSM
        (GZIPPY_DEBUG confirms the renamed contract; was IsalParallelSM).
  - NOTE: env lacks a subagent-spawn tool (only TaskStop surfaced); leader executed the
    mechanical gate directly. Phase 2 design/impl/profiling will need the spawn tool —
    flag to supervisor if delegation is required there.

## PHASE 1 — DONE. PHASE 2 — IN PROGRESS.

### STEP 0 [DONE 2026-06-06] — rename integrated onto working branch
- `git merge --ff-only phase1-naming-truth` -> reimplement-isa-l now at 6b7d6f5 (linear ff).
- VERIFIED: StreamingInflateWrapper present (7+ src files); 8 remaining IsalInflateWrapper
  refs are ALL intentional vendor doc-citations (rapidgzip::IsalInflateWrapper / gzip/isal.hpp
  in backends/isal_decompress.rs, inflate/resumable.rs, inflate/staged_bits.rs,
  parallel/inflate_wrapper.rs) — kept for traceability per approved MAP. Renamed files
  lut_bulk_inflate.rs + lut_huffman.rs present; old isal_* names gone.
- Build clean: release --no-default-features --features pure-rust-inflate, 34s, warnings only.
- pgrep NOTE: bare `pgrep -f 'cargo|rustc'` FALSE-POSITIVES on Cursor helpers/shells whose
  cwd contains "isal". Use `pgrep -x cargo; pgrep -x rustc` for the build-serialization gate.

### STEP 1 [DESIGN DONE — advisor-review pending] — two flag-gated paths
DESIGN SUBAGENT (exit 0, ~325s) findings + leader-corroborated:
- KEY: the unified pure-Rust path ALREADY EXISTS and is the SOLE decode path. NO live
  ISA-L decode anywhere in parallel/+inflate/ (build.rs:80-82 pure_inflate_decode==parallel_sm).
  => gzippy-native = today's pure path made default (STEP 1 trivial wiring). gzippy-isal =
  RE-INTRODUCE a real-ISA-L clean tail behind a flag (the path needing new code).
- CAVEAT (leader-verified): gzippy-isal FFI building block EXISTS and is tested:
  isal_decompress.rs:307 decompress_deflate_from_bit (+_with_end:460, +_with_boundaries:643)
  — bit-offset + 32KB-dict tests pass.
- CORRECTION (advisor caught my error): the stopping-point patch IS present — in the C
  SUBMODULE vendor/isa-l/igzip/igzip_inflate.c:1412,1747 + header igzip_lib.h:232 ("gzippy/
  rapidgzip patch") + bindings isal-sys/src/igzip_lib.rs:1565-1569 (full END_OF_BLOCK[_HEADER]/
  END_OF_STREAM[_HEADER] enum). My earlier "no patch" grep checked the Rust wrapper, not the C
  submodule — WRONG. So a faithful tail with REAL block boundaries is available.
- Today's chunk decode: marker phase (marker_inflate::Block, 128KiB u16 ring) -> FlipToClean at
  clean_appended_len>=MAX_WINDOW_SIZE (gzip_chunk.rs:1191, mirrors GzipChunk.hpp:521) ->
  finish_decode_chunk_impl (gzip_chunk.rs:354) constructs StreamingInflateWrapper (pure-Rust
  Inflate<Clean,Generic,Streaming> -> ResumableInflate2, a SECOND 128KiB-staging engine).
- CACHE MANDATE GAP: "one engine, flip-in-place" NOT yet met — TWO engines/buffers per chunk
  (~288KiB scratch). That fold is the substantive native work (sequenced last, A/B-gated).
- WIRING PLAN: gzippy-native=["pure-rust-inflate"]; gzippy-isal=["pure-rust-inflate",
  "isal-compression"]; new build.rs cfg isal_clean_tail = is_x86_64 && CARGO_FEATURE_GZIPPY_ISAL.
  KEEP default=[] through STEP 1+2 (flipping surfaces parallel-module lint debt per Cargo.toml:46-48);
  flip to gzippy-native after STEP 2 clears debt. arm64: gzippy-isal degrades to pure tail (==native).
- SEQUENCE: 1a wiring(zero-behavior, native==today byte-identical) -> 1b isal tail (x86_64) ->
  STEP2 delete dead (route_c_*, not-parallel_sm stub CONFIRMED; consume_first_decode big FASTLOOP
  fns + jit_decode/specialized_decode/double_literal + MarkerRing NEEDS-DEEPER-CHECK) ->
  native cache work (share fixed tables / pool engines / fold-to-one-engine / BMI2 runtime dispatch).

### ADVISOR VERDICT (exit 0, ~275s): APPROVE-WITH-CHANGES. Strategy + native sound. 4 fixes
incorporated into the plan (all leader-verified):
- FIX1 native drops ISA-L *compression* backend: release/CI ship --features isal-compression
  (release.yml:132) giving x86 T1 L0-L3 ISA-L COMPRESS; gzippy-native=["pure-rust-inflate"]
  loses it. FFI-free mandate is DECODE-ONLY. DECISION: gzippy-native keeps isal-compression for
  COMPRESS on x86_64 only (decode stays pure-Rust/FFI-free). Update Makefile + .github/workflows
  to new feature names so the gated path == what ships.
- FIX2 arm64 C-build: isal-sys/build.rs has NO arch guard (verified: handles riscv64/win/macos,
  no x86 gate) -> gzippy-isal pulling isal-compression would try to build ISA-L C on arm64. Add
  build.rs compile_error! when gzippy-isal && !x86_64. gzippy-isal = x86_64-only ref baseline.
- FIX3 add CI line for the new combo pure-rust-inflate+isal-compression (not built today).
- FIX4 (BIGGEST RISK) the isal tail is NOT at parity just because _with_boundaries records
  boundaries. Divergences: pure tail coalesce-stops at first boundary AT-OR-PAST stop_hint_bits
  (gzip_chunk.rs:387) & commits CRC only up-to-boundary bytes (:507-516); isal fns run to
  BFINAL over whole slice. Stopping-point set differs (isal _with_boundaries requests only
  END_OF_BLOCK :666 vs pure END_OF_BLOCK|_HEADER|STREAM_HEADER :382-384). Output: pure STREAMS
  into writable_tail segment-by-segment; isal returns one Vec (needs append+offset glue,
  mirror :428). Two stop modes until_exact true/false (:374). VERIFY-FIRST GATE for 1b: a
  same-input pure-vs-isal differential on real silesia chunks asserting IDENTICAL (decoded
  bytes, committed length, end_bit handoff, per-chunk CRC) before any production wiring.
- Keep default=[] through STEP1+2; silesia gate ALWAYS runs explicit --features gzippy-native
  (bare cargo compiles no decode path: routing tests are #[cfg(parallel_sm)]).

### STEP 1a [DONE 2026-06-06] — zero-behavior wiring
- Cargo.toml: gzippy-native=["pure-rust-inflate"]; gzippy-isal=["pure-rust-inflate",
  "isal-compression"]. build.rs: cfg isal_clean_tail = is_x86_64 && CARGO_FEATURE_GZIPPY_ISAL
  && parallel_sm; panic!() if gzippy-isal && !x86_64 (arm64 C-build guard, FIX2).
- GATE (--features gzippy-native): build clean 34s (same 6 pre-existing warnings as
  pure-rust-inflate => pure alias). silesia sha 028bd002...410f == reference via path=ParallelSM.
  lib tests 35 passed / 0 failed (routing:: + pure_rust_inflate_corpus::, --test-threads=1,
  load-flaky skipped). Native is byte-identical to today's decode, by construction.
- No isal_clean_tail code arm yet (that's 1b). NEXT: 1b verify-first differential gate then tail.

### STEP 1b [IN PROGRESS — impl subagent running, builds x86_64 via Rosetta]
- Rosetta prereqs VERIFIED: target x86_64-apple-darwin installed; arch -x86_64 uname -m == x86_64.
  gzippy-isal builds ONLY on x86_64 (build.rs panics on arm64) so Rosetta is mandatory for it.
- Subagent task: (DELIVERABLE 1, the GATE) a #[cfg(isal_clean_tail)] differential test on REAL
  silesia chunks asserting pure-tail vs isal-tail parity on (decoded bytes, committed length,
  final_bit handoff, per-chunk CRC, block boundaries) for BOTH until_exact true/false; report
  exact divergence mechanism if any. (DELIVERABLE 2, only if parity achievable) the
  #[cfg(isal_clean_tail)] arm in finish_decode_chunk_impl. Build-serialized (pgrep -x), no commit.
- IMPORTANT: leader must NOT start any build while this subagent runs (it owns the build lock).

### STEP 1b [GATE DONE + verdict — committed 8d026a8] — faithful isal tail DEFERRED to last
GATE RESULT: differential PASSES 10/10 (5 silesia chunks x until_exact{T,F}) — ISA-L tail
records byte-identical (committed bytes, length, final_bit, CRC, boundaries) vs pure tail.
Existence proof a faithful gzippy-isal tail is achievable. Test-only, cfg(all(test,isal_clean_tail)).
NOTE: arch -x86_64 cargo IMPOSSIBLE (cargo is arm64-only binary); the working x86_64 path is
NATIVE cargo cross-compile --target x86_64-apple-darwin with CARGO_BUILD_RUSTFLAGS="-C
target-cpu=x86-64" (override .cargo/config.toml target-cpu=native), runs under Rosetta.

ADVISOR VERDICT (2nd advisor, independent rewind-logic verification): 
- Design (A) CORRECT (swap inner engine for interface-compatible ISA-L wrapper, driver loop
  unchanged, accounting matches by construction incl. the :478-489 rewind). Design (B)
  decode-to-end+reselect is UNFAITHFUL by construction (handoff bit is driver-loop-history-
  dependent: :486 rewinds to last_eob_pos recorded at :475 on the PRIOR EOB). No thinner shim.
- SEQUENCING (the real call): STEP 2 (delete dead) -> gzippy-native cache work -> gzippy-isal
  Design-A tail LAST, immediately before PHASE 3. WHY: isal is an INSTRUMENT not a deliverable;
  native's fold-to-one-engine WILL REWRITE the driver interface Design-A swaps into (build isal
  now => re-port; build last => one port); gate 8d026a8 already proves pure≡isal so native
  doesn't need isal as oracle (native gated by flate2+libdeflate+canonical sha, arm64-portable);
  defer the riskiest C-FFI re-add until PHASE 3 actually consumes it.
- CONSTRAINT: STEP 2 must NOT delete finish_decode_chunk_impl (live prod + Design-A insertion
  point). native fold should cfg-FORK it (#[cfg(isal_clean_tail)] keeps two-phase; native folded),
  NOT delete. Delete only the genuinely-dead alts (route_c_*, jit_decode, specialized_decode,
  double_literal, not-parallel_sm stub) — advisor-gate each.
- BEFORE any Design-A impl: harden the gate to exercise the :486 rewind (a non-final/non-fixed
  rewind case; a fixed-Huffman no-rewind case per :481 not_fixed; the ==stop_hint case) AND
  verify ISA-L's bit_position at END_OF_BLOCK[_HEADER] matches the pure tell_compressed()
  convention. Q3+Q4 are the SAME gate.

## STEP 2 — DELETE DEAD INFLATE IMPLS [IN PROGRESS 2026-06-06]
### Deletion MANIFEST produced (read-only subagent, exit 0) — far more conservative than expected
- KEY FINDING: a SECOND live decode path (scan_inflate.rs -> decompress::index -> index_mode.rs ->
  CLI --index/--seek, cli.rs:212) compiles under BOTH features and keeps jit_decode,
  specialized_decode, double_literal, bmi2, libdeflate_entry, consume_first_decode ALL LIVE.
  bgzf.rs inflate_consume_first calls are #[cfg(test)]-only (not a live anchor).
- NO dead parallel/ module found — every #[cfg(parallel_sm)] module has a real chunk-pipeline
  consumer (decode_bypass LIVE @ sm_driver:64/chunk_fetcher; marker engine LIVE; etc).
- Only TRULY dead: ONLY route_c_dynasm.rs + route_c_fixed.rs (gated on feature route-c-dynasm
  NOT pulled by default/pure-rust-inflate/gzippy-native/gzippy-isal/tests; only mod.rs:25-28 refs).
- PARTIAL: libdeflate_decode.rs — KEEP get_fixed_tables (live @ resumable.rs:951); dead big fns
  (decode_libdeflate, inflate_libdeflate, copy_match, file-local BitReader, decode_with_double_cache
  + their in-file #[cfg(test)] tests 979-1196) have ZERO external callers.
- PROPOSED BATCHES: B1 route_c_* (zero prod/test impact); B2 dead libdeflate_decode fns;
  B3 (deferred, needs decode_dynamic dispatch trace) consume_first_decode internal prune +
  double_literal reclassification. MUST-PRESERVE verified: finish_decode_chunk_impl + STEP-1b gate.
- ADVISOR REVIEW [DONE, exit 0]: APPROVE Batch1; APPROVE-WITH-CHANGES Batch2 (manifest had 2
  WRONG symbol names: BitReader->LibdeflateBits; decode_with_double_cache->decode_huffman_double_lit;
  advisor enumerated the real dead cluster by reading the file). Both riskiest claims independently
  grep-confirmed. Gate adds: include plain cargo build/test; Batch2 gate must exercise --index/scan
  path (tests::index). Retire route-c-dynasm feature+deps+sub-crate = SEPARATE follow-up (not byte-neutral).

### BATCH 1 [DONE — committed 03f592e] delete route_c_dynasm.rs + route_c_fixed.rs + mod.rs:25-28
- DUAL-SHA gate GREEN: gzippy-native + gzippy-isal(Rosetta x86_64 cross-compile, --target
  x86_64-apple-darwin, CARGO_BUILD_RUSTFLAGS=-C target-cpu=x86-64) BOTH emit silesia sha
  028bd002...410f via path=ParallelSM. routing 30/0, correctness 130/0, pure_rust_inflate_corpus
  5/0 (incl silesia), index 22/0; plain cargo build clean. NOTE: native cargo build --features
  gzippy-isal on arm64 PANICS by design (build.rs:83 guard) — MUST use Rosetta cross-compile.

### BATCH 2 [DONE — committed edcd863] prune dead big-decoder cluster in libdeflate_decode.rs
- Replaced file with ONLY get_fixed_tables + FIXED_TABLES (was ~1200 lines). Removed the dead
  cluster (zero external callers under either feature). KEEP get_fixed_tables (live @
  resumable.rs:951 + consume_first_decode.rs + double_literal.rs).
- RESUMED after restart: prior leader's interrupted edit reconciled under the NEW global build
  mutex scripts/cargo-lock.sh (portable mkdir-lock; macOS has no flock). DUAL-SHA GREEN:
  gzippy-native (arm64) + gzippy-isal (Rosetta x86_64 cross-compile) both emit silesia
  028bd002...cb410f via path=ParallelSM. native suites 187/0 (routing/correctness/
  pure_rust_inflate_corpus/index, --test-threads=1, load-flaky skipped). clippy clean (fixed
  inherited needless_range_loop in get_fixed_tables, behavior-identical).
- STRUCTURAL FIX: ALL cargo/rustc/test now wrapped `scripts/cargo-lock.sh <cmd>` — concurrent
  builds structurally impossible. Every subagent MUST use it.

### BATCH 3 [DONE — committed 70ec5ff] dead double-literal + short-bits caches
- AFTER restart: read-only scout did the decode_dynamic dispatch trace (the thing that made B3
  ambiguous) + an independent advisor read corroborated. BOTH confirm decode_huffman_cf:1505 is
  LIVE (fixed-block + dynamic-block paths) and KEPT. Everything deleted has ZERO callers under
  gzippy-native, gzippy-isal, OR scan_inflate/index, plus tests/benches/examples.
- Deleted: double_literal.rs, huffman_short_bits_cached_deflate.rs, huffman_short_bits_multi_cached.rs
  (whole files + mod decls); 9 dead fns in consume_first_decode.rs (3580->2628 lines); stale
  doc-comment refs fixed. NOT touched: finish_decode_chunk_impl; specialized_decode cluster
  (production-dead but SpecializedCache coupled to SPEC_CACHE + test stats — coupled edit, deferred).
- DUAL-SHA gate GREEN (under scripts/cargo-lock.sh): native + isal(Rosetta x86_64) both
  028bd002...cb410f via path=ParallelSM; native suites 187/0; clippy clean; plain build clean.

## STEP 2 — DONE (B1 route_c 03f592e, B2 libdeflate_decode edcd863, B3 70ec5ff).
The ONLY remaining production-dead cluster (specialized_decode + SPEC_CACHE stats) is a coupled
edit, intentionally deferred — not a clean leaf delete.

## NATIVE CACHE FOLD — design + advisor DONE 2026-06-06; impl in progress
- DESIGN (subagent, read-only) + ADVISOR review (both this session). Two engines per chunk confirmed:
  Engine M = marker_inflate::Block (128KiB u16 ring; faithful port of vendor deflate.hpp deflate::Block,
  has BOTH markers + clean drain @ marker_inflate.rs:738-750). Engine C = StreamingInflateWrapper ->
  ResumableInflate2 (128KiB staging staged_bits.rs:22 + 32KiB window resumable.rs:94/107 +
  last_32kib_window_vec gzip_chunk.rs:767 ~= 192KiB UNPOOLED Box churn/chunk), constructed @
  finish_decode_chunk_impl gzip_chunk.rs:354/379 after FlipToClean gzip_chunk.rs:1191. Engine C cites
  gzip/isal.hpp — it is the pure-Rust IsalInflateWrapper STAND-IN (belongs to gzippy-isal path).
- FIXED Huffman tables ALREADY SHARED (FIXED_TABLES OnceLock libdeflate_decode.rs:17). Dynamic tables
  inherently per-block (reuse, not share). So mandate's "share tables" already met for the shareable one.
- SURVIVOR FORK RATIFIED (advisor + governing memory): Engine M survives, Engine C deleted from native.
  Engine M == MarkerRing engine the governing memory mandates flip-in-place + continue clean tail on
  SAME cursor; Engine C (IsalInflateWrapper stand-in) stays on gzippy-isal -> real ISA-L FFI Design-A.
- ADVISOR CAUGHT A BUG in the naive first step: pooling the WHOLE Engine C is NOT byte-exact —
  set_window:336 reset is INCOMPLETE (misses encoded_until_bits, coalesce_stop_hint [concrete stale-hint
  divergence: set only when !until_exact gzip_chunk.rs:386-388], block_boundaries) AND a lifetime
  soundness problem (ResumableInflate2<'a> borrows per-chunk input, can't live in 'static thread_local).
- CORRECTED STEP 1 (impl now): pool the TWO HEAP BUFFERS only (128KiB staging Box + 32KiB window Box)
  via thread-local free-list; engine logic + lifetimes UNTOUCHED, only the allocator changes. Near
  byte-transparent. Gate: dual-sha + STEP-1b per-chunk differential. Then STEP 2 flip-in-place fold
  (Engine M continues clean tail, delete Engine C from native) + clean-drain bulk/BMI2 graft IN THE
  SAME step (avoid shipping a wall regression; cfg-fork finish_decode_chunk_impl so isal keeps two-phase).

### NATIVE CACHE STEP 1a [DONE — committed 736ea1a] pool 128KiB staging box
- Scoped to the STAGING box only (the larger, always-fully-overwritten one — provably byte-transparent:
  recycled box's bytes always overwritten by first reload_at_bit before any read). STAGING_POOL
  thread-local free-list (cap 4); buf is ManuallyDrop so Drop recycles instead of frees (no double free).
- DUAL-SHA GREEN: native + isal(Rosetta x86_64) both 028bd002...cb410f via path=ParallelSM. staged_bits
  unit tests 7/0 (reload/memcpy parity on pooled box); native suites 187/0; clippy clean.
- Hit + fixed a test-only E0507 (a unit test moved *staged.buf; now .to_vec()/.as_slice() through MD).
- STEP 1b TODO: 32KiB SlidingWindow pool — DEFERRED, needs explicit reset analysis (read-before-fully-
  written within a chunk, unlike staging). May fold into the STEP-2 flip instead (Engine C goes away).
- TODO: pin the ~ footprint numbers EXACTLY (mandate is measured, not asserted) — build an RSS-vs-T
  instrument before the STEP-2 behavioral fold (advisor: validate the instrument on this predictable
  per-chunk->per-thread delta first).

## (orig) NEXT: STEP 2 — delete dead inflate impls (advisor-gate each). DO NOT touch finish_decode_chunk_impl.
- Features today: default=[] ; pure-rust-inflate (prod decode, sets cfg parallel_sm==
  pure_inflate_decode in build.rs:81-83) ; isal-compression (TRUE ISA-L *compression* only,
  no decode) ; isal=[] (build.rs-set when static lib built). NO live ISA-L decode currently.
- New features to add: gzippy-isal (faithful: marker engine -> REAL ISA-L FFI decode handoff
  at 32KiB clean) and gzippy-native (DEFAULT prod: one unified pure-Rust engine, cache-resident
  per design mandate). Design subagent spawned to produce the two-column rapidgzip<->gzippy map
  + concrete wiring plan before any impl.

## RSS-vs-T INSTRUMENT [DESIGN 2026-06-06 — leader; advisor-corroborate then delegate impl]
Advisor-mandated prerequisite before the STEP-2 behavioral fold. Measures the cache mandate
(plans/gzippy-native-design-mandate.md): shared small hot working set, RSS ~flat as T rises.
RSS captured LOCALLY (arm64 gzippy-native); wall/MPKI stay on neurotic locked harness.

### What it must measure (per binary, per corpus)
1. PEAK total RSS vs T in {1,8,16}: /usr/bin/time -l "maximum resident set size" (bytes on macOS).
2. STEADY-STATE RSS time-series: background `ps -o rss= -p <pid>` poller @ ~50ms -> median of
   the plateau (excludes ramp/teardown) so per-thread growth is separable from one-shot allocs.
3. Derived PER-THREAD working-set slope: (RSS@T16 - RSS@T1)/15 and (RSS@T8-RSS@T1)/7; report both
   (non-linearity is signal). Mandate target: slope << per-thread L2; RSS roughly flat in T.
4. sha256 of decoded output EVERY run == 028bd002...cb410f (a low-RSS run with wrong bytes is void).

### Validation (instrument must earn trust BEFORE its numbers count — CLAUDE.md rule 4/PROCESS)
- POSITIVE control: a knob that PROVABLY inflates per-thread RSS must show up as increased slope
  (e.g. force a larger per-thread alloc, or compare a high-T run that we KNOW duplicates buffers).
- NEGATIVE/self control: same binary vs itself reads equal RSS within poller spread (define spread).
- POOLING DELTA (the real validation): build 70ec5ff (UNPOOLED staging) vs 736ea1a (POOLED staging),
  run both through the instrument at T8/T16. Hypothesis: pooled shows LOWER per-thread RSS slope
  (recycled staging box => fewer live 128KiB allocs at steady state). MEASURE — if peak RSS is
  unmoved (pool may only cut alloc *rate*, not peak footprint), SAY SO; that's a finding about what
  the instrument can/can't see, not a failure. Either way the +/- controls must pass for trust.

### Deliverable
scripts/bench/rss_vs_t.sh — args: BIN, GZ corpus, T-list, N trials; outputs the table above +
controls. ALL builds via scripts/cargo-lock.sh (global mutex). Per-T binary = same gzippy-native
release binary, T set by `-p <T>`. Report per-T peak+plateau RSS, slope, sha-OK, control verdicts.

### ADVISOR VERDICT on RSS instrument design: APPROVE-WITH-CHANGES (incorporated)
KEY REFRAME: process RSS is TOO COARSE for the mandate. Per-thread decode scratch at T16 ~=
128KiB*16 = 2MB against a ~300-400MB backdrop (68MB input mmap + ~211MB decoded silesia) = <1%,
below poller jitter. RSS-vs-T CANNOT resolve per-thread decode buffers. So:
- PRIMARY mandate instrument = DIRECT IN-PROCESS BYTE ACCOUNTING (zero noise floor), via GZIPPY_DEBUG
  counters: peak live bytes per thread-local pool, sum of shared read-only table bytes, count of
  distinct pool high-water boxes. Measures "tiny per-thread working set / shared tables / pooled
  buffers" EXACTLY. This is what honors the mandate.
- RSS-vs-T DEMOTED to a coarse T-scaling GUARD: catches only UNEXPECTED large T-scaling (e.g. a
  per-thread COPY of a shared table — large enough RSS would see it). Necessary-not-sufficient.
- cache-residency is NOT an RSS question -> MPKI (deferred to neurotic locked harness) is the real test.
FIXES incorporated:
- UNITS: /usr/bin/time -l max-RSS is BYTES on macOS; ps -o rss= is KiB. Normalize + assert (1024x trap).
- Stream output to /dev/null (bounded writer) so output buffering is T-invariant + small.
- time -l for PEAK only; ps plateau-median for live set; NEVER cross-compare the two as equal.
- POOLING DELTA = a FINDING, not a control (free-list may not move peak live set; null is ambiguous).
- POSITIVE CONTROL (must-pass, calibrated/signed/magnitude): env knob allocs+TOUCHES (writes every
  page) known N MiB per worker thread; instrument MUST recover slope ~= N at multiple N (linearity).
- NEGATIVE control: binary vs itself within poller spread (keep).
- T grid {1,2,4,8,16}, REGRESS slope (2-pt slope can't separate const overhead from per-thread growth);
  check linearity not endpoints. N>=7 interleaved, Delta<spread => TIE.
- Assert GZIPPY_DEBUG path=ParallelSM (native engine, not fallback) every run; sha gate every run.

### RSS instrument — REVISED deliverables (to delegate)
D1 (PRIMARY): in-process per-thread byte accounting behind GZIPPY_DEBUG (or a new env GZIPPY_MEM_STATS=1):
   instrument STAGING_POOL (+ later the 32KiB window pool / engine state) to record peak-live-bytes per
   pool + box high-water count; emit shared-table byte total (FIXED_TABLES etc). NEEDS a small src change
   (counters) — design subagent first maps WHERE to hook, advisor-gate, then impl. Byte-exact (counters only).
D2 (GUARD): scripts/bench/rss_vs_t.sh — peak (time -l, bytes) + plateau (ps, KiB) RSS over T{1,2,4,8,16},
   N>=7 interleaved, output->/dev/null, sha gate, native-path assert, slope regression + linearity.
D3 (CONTROL): GZIPPY_MEM_BALLAST_MIB=N env -> each worker allocs+touches N MiB; rss_vs_t must recover
   slope~=N at N in {8,16,32}. Pooling delta (70ec5ff vs 736ea1a) run + reported as a FINDING.

### RSS instrument — IMPL NOTE (impl subagent 2026-06-06) — exact hook lines + byte-transparency
New module `src/decompress/inflate/mem_stats.rs` (pub mod in inflate/mod.rs). COUNTERS ONLY, no decode
behavior change. `enabled()` reads `GZIPPY_MEM_STATS` ONCE into a OnceLock<bool>; every hook is
`if !enabled() { return; }` so the flag-off path is an inlined early-return (DUAL-SHA gate proves
byte-identical on/off).
HOOKS:
- staged_bits.rs take_staging_box (~35): split pop into reused vs Box::new arm, call
  `mem_stats::on_take(reused)`. on_take: relaxed-atomic alloc-vs-reuse counter; thread_local live++/peak;
  on a new per-thread peak records into a global Mutex<HashMap<ThreadId,ThreadStat>> (cold path).
- staged_bits.rs return_staging_box (~42): `mem_stats::on_return()` (thread_local live--). Drop path
  already calls return_staging_box (line 329) so per-chunk lifecycle is covered.
- staged_bits.rs STAGING_POOL_CAP -> env-overridable `pool_cap()` OnceLock (default 4); CAP=0 disables
  pooling. This is the chosen D3 POOLING-DELTA mechanism (env `GZIPPY_STAGING_POOL_CAP=0` vs 4 on the
  SAME 736ea1a binary) — isolates the pooling variable with one build, no worktree/cherry-pick drift.
  Default 4 == today's production behavior (byte- AND perf-identical when env unset).
- libdeflate_entry.rs: add `heap_bytes()` to LitLenTable + DistTable (size_of::<Self>() +
  capacity*size_of::<Entry>()) so report() prints the ONE shared-table footprint via get_fixed_tables().
- main.rs main() after `let result = run()`: `decompress::inflate::mem_stats::report();` BEFORE
  process::exit (which skips destructors). report() is itself enabled()-gated.
POSITIVE CONTROL: `GZIPPY_MEM_BALLAST_MIB=N` (own OnceLock switch, independent of MEM_STATS so the RSS
guard drives it without accounting overhead). ensure_ballast() on first take per thread allocs+touches
(4KiB stride) N MiB into a thread_local Vec held for thread life -> known per-thread resident slope.
Byte-transparent: ballast Vec is never read by decode.

## FLIP-IN-PLACE FOLD — leader orientation (gated on RSS instrument; STEP 2 next)
Read the two-engine boundary in gzip_chunk.rs + marker_inflate.rs:
- Engine M loop (run_marker_blocks-style, gzip_chunk.rs:1184-): on clean_appended_len>=MAX_WINDOW_SIZE
  && !ctx.flipped it returns MarkerStep::FlipToClean{end_bit_offset, window_len=32KiB} at :1191-1202,
  setting ctx.current_bit_offset=next_block_offset. Driver THEN constructs Engine C
  (StreamingInflateWrapper -> ResumableInflate2) in finish_decode_chunk_impl (:354) to decode the
  clean tail from that bit offset. THAT handoff is the two-engine cost the fold removes (native only).
- Engine M ALREADY CAN drain clean u8 in-place: marker_inflate.rs:731-758 drain_to_output ->
  push_clean_u8 (:750) once contains_marker_bytes==false (vendor deflate.hpp:1285-1292). So the fold =
  do NOT return FlipToClean in native; keep iterating Engine M's block loop on the SAME ctx cursor,
  letting drain_to_output emit clean bytes, until BFINAL/stop_hint. Engine C deleted from native;
  finish_decode_chunk_impl cfg-FORKED (#[cfg(isal_clean_tail)] keeps two-phase for gzippy-isal Design-A;
  native folded) — advisor CONSTRAINT: do NOT delete finish_decode_chunk_impl (Design-A insertion point).
- Gate: DUAL-SHA 028bd002...cb410f (native folded + isal two-phase), per-chunk differential (STEP-1b gate
  8d026a8 style), then measure wall+RSS on locked harness. Sequence the clean-drain bulk/BMI2 graft IN
  THE SAME behavioral step to avoid shipping a transient wall regression.

## LEADER RE-DRIVE 2026-06-06 (resumed @ 5f162bb) — singleton-leader discipline adopted
- SINGLETON-LEADER LOCK created scripts/leader-lock.sh (clone of cargo-lock mkdir-mutex + stale-pid
  reclaim; acquire/release/status verbs; NON-BLOCKING acquire => duplicate leader EXITS). Self-test
  PASSED: positive (reclaims stale pid 99999), negative (refuses while live holder). ACQUIRED by a
  persistent sentinel pid (nohup sleep) so the lock survives across leader turns (a Bash-subshell
  $$ dies between calls => would read stale).
- FOLD insertion point CONFIRMED by leader read: the ONLY behavioral change is at
  marker_decode_step_loop gzip_chunk.rs:1191-1202 (the `clean_appended_len()>=MAX_WINDOW_SIZE
  && !ctx.flipped` FlipToClean early-return). cfg-fork it: #[cfg(isal_clean_tail)] keeps the
  FlipToClean return (gzippy-isal two-phase, Design-A insertion intact); #[cfg(not(isal_clean_tail))]
  (native) sets ctx.flipped=true and CONTINUEs the loop. Engine M's read() already drains clean u8
  in-place (marker_inflate.rs:1011 drain_to_output -> push_clean_u8 once contains_marker_bytes==false);
  UnifiedMarkerSink.push_clean_u8 buffers to pending_clean, flushed to chunk.data each step
  (decode_chunk_unified_marker :744-749). Loop terminates at BFINAL/stop_hint via MarkerStep::Finished.
  finish_decode_chunk_impl UNTOUCHED (still reachable on isal + window-seeded path). cfg name confirmed
  build.rs:101 isal_clean_tail = is_x86_64 && gzippy-isal && parallel_sm.
- HAZARD to gate: ring-overwrite if a single post-flip clean block is huge and drain only fires at
  EOB (marker_inflate.rs:725-730 contract). read_internal_compressed must drain mid-block OR the
  differential will catch it. The per-chunk differential is the verdict.

## FOLD COMMITTED [DONE 2026-06-06 — 8cfad3a] flip-in-place one-engine fold
- COMMIT 8cfad3a on reimplement-isa-l (parent 5f162bb). Only src change: gzip_chunk.rs +322/-11.
- STEP-5 "FAIL" ROOT CAUSE (corroborated, NOT flaky, NOT a regression): the fold-gate STEP-5 ran
  `cargo test ... --lib routing correctness pure_rust_inflate_corpus -- ...` — `cargo test` accepts
  only ONE positional TESTNAME, so it errored `unexpected argument 'correctness'` and RAN ZERO TESTS.
  Neither the project_parallel_test_hang deadlock-flake NOR a byte regression. Leader rerun under
  scripts/cargo-lock.sh with the supervisor's valid invocation (full --lib, --test-threads=1, 4
  excludes) = 850 passed / 0 failed, matching the supervisor exactly. Fold is byte-exact-clean.
- VALIDATION (all green for the committed tree): DUAL-SHA native arm64 + isal x86_64(Rosetta) both
  028bd002...cb410f via path=ParallelSM; native_fold_parity 32/32 correct (12 FLIPPED; rewind /
  fixed-Huffman / ==stop_hint; until_exact {T,F}); lib suites 850/0.
- GATE HARDENED: /tmp/fold-gate.sh STEP-5 now uses the supervisor's invocation — full `--lib`,
  `--test-threads=1`, excludes by EXACT name (diff_ratio_parallel_single_member_speedup,
  scoped_cancel_stops_early_without_full_scan, hot_path, alloc_budget). No multi-positional filter.
- EXCLUDE/FLAKY POLICY (board record): the 4 names above are the canonical load-flaky/perf-gate
  exclusion set for serial byte-exact lib runs. ALWAYS pass `--test-threads=1` (project_parallel_test_hang
  deadlock-class). Use full `--lib` + `--skip <exact-name>` — NEVER multiple positional TESTNAME filters
  (silent CLI parse error that runs nothing and reads as a FAIL).
- ADVISOR GAP (assumption, flagged): no subagent-spawn tool in this session, so a fresh Opus advisor
  could not be consulted for the commit. Proceeded on standing advisor verdicts already on record (fold
  design APPROVE lines 163-181; finish_decode_chunk_impl-preserve constraint satisfied) + the direct
  flaky-vs-real resolution above. If a supervisor advisor pass is required retroactively, the evidence
  is captured here and in /tmp/fold-gate.result.

## NEXT (sequenced): BMI2 PEXT/BZHI + multi-symbol LUT graft into Engine M
- Cache mandate: ONE shared decode-table copy across threads, no large per-thread buffers, RSS ~flat in T.
- Each technique byte-exact + measured on the locked harness; keep wins, revert regressions.
- Then: gzippy-isal Design-A tail (dual-sha vs folded driver) -> PHASE 3 3-way Fulcrum + RSS/working-set/MPKI.

## SPEED STRETCH — LEADER RE-DRIVE 2026-06-06 (resumed @ 8cfad3a). leader-lock HELD.
### FALSIFIER PRE-REGISTERED [DONE] plans/bmi2-graft-falsifier.md
- Per-technique falsifier + ceiling + TIE/regression judgment recorded BEFORE any opt work.
- KEY PRIOR-STATE FINDING (read the production loop, marker_inflate.rs run_multi_cached_loop
  :1630+): techniques (b) multi-symbol LUT and (c) lean refill are ALREADY PRESENT
  (2-/3-literal speculative chain :1782-1841; branchless bounds-elided refill_fast :1715-1731,
  REFILL_THRESHOLD=48; speculative next-entry carry). Only (a) BMI2 PEXT/BZHI is genuinely
  un-grafted (Generic HAS_BMI2=false unified.rs:122; no BMI2 in the marker hot loop). So (a)
  is THE lever; (b)/(c) are re-validation/sharpening (re-attempt ca52389-class with fresh
  measurement — KNOWN HAZARD).
- unified.rs is a DELEGATING SCAFFOLD (Inflate<Clean,Generic,Streaming> -> ResumableInflate2);
  the live production dynamic-Huffman hot loop is marker_inflate.rs read_internal_compressed_
  canonical_specialized -> run_multi_cached_loop (uses libdeflate LitLenTable/DistTable).
  The fixed-Huffman arm uses HuffmanCodingReversedBitsCached. THAT is the BMI2 graft surface.

### INSTRUMENT-VALIDITY FINDING (must fix before cache-mandate measurement)
- The byte-accounting instrument (mem_stats.rs, hooked to staged_bits.rs take_staging_box) is
  DEAD on the native path AFTER THE FOLD. take_staging_box is only called by StagedBits (part of
  ResumableInflate2 = Engine C); the fold removed Engine C from native (finish_decode_chunk_impl
  unreached in native steady state). Result: GZIPPY_MEM_STATS=1 on native silesia emits NO report
  (threads observed = 0) — leader verified locally.
- The ACTUAL native per-thread working set is BOOTSTRAP_BLOCK (thread_local Block, gzip_chunk.rs:1096),
  dominated by Block.output_ring: Box<[u16; RING_SIZE]> = 2*MAX_WINDOW_SIZE = 128KiB per worker
  thread. The instrument must be RE-HOOKED to this (the ring + literal_cl/backreferences Vecs +
  the shared FIXED_TABLES bytes) before it can measure the cache mandate. This re-hook is byte-exact
  (counters only) and needed for PHASE 3's RSS/working-set numbers; it does NOT block the BMI2 graft
  go/no-go (that's a wall question, answered by the locked harness).

### CEILING-BOUNDING METHOD — open question for advisor (flag to supervisor)
- A clean DECODE-ZERO oracle (charter rule 3 "remove the region, measure") is NOT byte-exact here:
  zeroing the inner decode produces wrong bytes (can't sha-verify), violating the byte-exact ABSOLUTE
  invariant. So the textbook removal-oracle is unavailable for the inner Huffman loop.
- Available byte-exact bounds: (1) the existing GZIPPY_SLOW_* decode/bootstrap slow-injection
  (already moved the wall ~proportionally, survived freq-neutral disproof) gives the SLOPE / confirms
  criticality; (2) the BMI2 graft delta itself IS a perturbation (changes per-symbol decode cost by a
  known mechanism; read the interleaved wall response). Plan: confirm criticality + quantify slope via
  SLOW-injection on the locked harness (small factors, freq-neutral control), THEN graft+measure;
  Δ>spread=lever; TIE=keep-if-byte-exact-no-RSS-regress; regression=revert.
- ADVISOR ASK (supervisor, you run advisors I can't): is the SLOW-injection-slope + graft-delta the
  acceptable ceiling-bound given a byte-exact decode-zero oracle is impossible, or is there a
  byte-exact removal-oracle shape I'm missing (e.g. swap-in a precomputed correct output buffer so the
  decode loop is bypassed but bytes are still correct)? This is the only consequential method call.

### *** PIVOTAL FINDING: BMI2 IS ALREADY ON in the measured locked-harness build ***
The charter's leading lever (a) "BMI2 PEXT/BZHI runtime dispatch (currently OFF, unified.rs:122)"
rests on a FALSE premise for the perf-target build. Evidence (all leader-verified this session):
- The production BMI2 path is NOT in unified.rs (a DEAD delegating scaffold). It is in bmi2.rs
  (extract_varbits :109-118, decode_extra_bits :78-97 — the extra-bits extraction) +
  consume_first_decode.rs::bzhi_u64 :168 + two_level_table.rs :353. ALL gated
  `#[cfg(all(target_arch="x86_64", target_feature="bmi2"))]` → emit BZHI when the feature is on.
- libdeflate_entry.rs decode_length :224 / decode_distance :322 (THE production dynamic-Huffman hot
  loop extract, run_multi_cached_loop) call bmi2::extract_varbits → already BZHI on bmi2 builds.
- THE BUILD ENABLES IT: .cargo/config.toml [build] rustflags=["-C","target-cpu=native"] AND the
  guest harness scripts/bench/guest_fulcrum_capture.sh:67 export RUSTFLAGS=-C target-cpu=native.
  On the BUILD GUEST (root@REDACTED_IP, where cargo build actually runs — neurotic login shell has no
  rustc on PATH), `rustc --print cfg -C target-cpu=native` emits target_feature="bmi2" (+avx2,
  pclmulqdq, vpclmulqdq); /proc/cpuinfo shows bmi2. ⇒ EVERY existing locked-harness wall number was
  measured WITH BMI2 BZHI ACTIVE. "Turning BMI2 on" is a no-op on the measured path.
- has_bmi2() (the runtime detector, bmi2.rs:26) is DEAD — never gates the hot path; the path is
  compile-time. So "runtime dispatch" only matters for a PORTABLE binary (built WITHOUT target-cpu=
  native, dispatching at runtime) — which is NOT the perf target (CLAUDE.md "arch-specific is the
  target; portable ships later via runtime dispatch").
- The ONLY genuinely-ungrafted BMI2 op is PEXT (_pext_u64) — ZERO uses in src. But the table index
  is a single masked field (litlen.lookup / dist.lookup), where BZHI/AND already wins; PEXT pays only
  for SCATTERED multi-field extraction, which this loop does not do. No obvious PEXT lever.
- CEILING-BOUND (the perturbation, not extrapolation): build native WITH vs WITHOUT bmi2 on the guest
  and measure the interleaved wall delta on the locked harness. That delta IS the BMI2 ceiling.
  Hypothesis: small/TIE (BZHI vs shift+mask is ~1-2 cycles on one extract per packet, against a
  461ms wall). Running this A/B next — it is the verdict, byte-exact (both correct), no extrapolation.

### BMI2 CEILING-BOUND A/B — DONE [VERDICT: TIE, lever (a) REJECTED with mechanism]
Ran scripts/bench/bmi2_ceiling_ab.sh on build guest 199 (silesia-large 162MB, T8 mask 0-7,
interleaved best-of-9). Both arms byte-exact (sha e114dd2b... == gzip ref) via path=ParallelSM.
- BMI2-ON  (target-cpu=native, PRODUCTION DEFAULT): best 0.6045  median 0.6485  mean 0.6526  σ 0.0316
- BMI2-OFF (target-cpu=native -C target-feature=-bmi2): best 0.6337 median 0.6484 mean 0.6588 σ 0.0235
- delta(best) = 29ms/4.6% but delta(MEDIAN) = -0.1ms / -0.02% (a DEAD TIE). within-arm spread
  63-108ms, σ 24-32ms — both >> any signal. The best-of-9 "4.6%" is an ON-side fast outlier; the
  robust statistic (median) is identical. Δ << spread ⇒ TIE, full stop.
- MECHANISM (a rejection per CLAUDE.md rule 7a, not a narrow miss): BMI2 BZHI is ALREADY compiled
  into the measured production binary (target-cpu=native enables target_feature=bmi2 on the guest
  CPU; extract_varbits/decode_extra_bits/bzhi_u64 emit BZHI). The "runtime-dispatch graft" would
  change NOTHING on the perf-target build — it only matters for a PORTABLE binary (not the perf
  target). And even forcing BZHI fully OFF is a wall TIE: the single extra-bits extract per packet
  is invisible against the ~600ms wall, which is memory-bound (128KiB u16 ring + 32KiB window apply
  + back-ref copies), NOT per-symbol ALU. rapidgzip's existence-proof advantage is therefore NOT in
  this op — it must be elsewhere (window apply / ring layout / copy), which is the cache-mandate
  surface, not BMI2.
- CONSEQUENCE: do NOT graft BMI2 runtime dispatch into Engine M for perf (no-op on perf build;
  the only beneficiary, a portable binary, is explicitly deferred per CLAUDE.md "portable ships
  later"). Techniques (b) multi-symbol LUT + (c) lean refill are ALREADY PRESENT (falsifier finding)
  AND are also per-symbol-ALU techniques the SAME memory-bound argument covers — re-validation is
  unlikely to move the wall by the same mechanism. The campaign's real lever is the cache mandate
  (shared tables already done; per-thread 128KiB ring is the next surface), measured by MPKI on the
  locked harness — NOT the ISA-L per-symbol hot-technique graft.
- ADVISOR CORROBORATION NEEDED (supervisor — you run advisors I can't): this REJECTS the charter's
  leading lever (a) with a mechanism. Please corroborate (1) the BMI2-already-on finding, (2) the
  TIE verdict (median not best-of-N), (3) the redirect from per-symbol-ALU grafts to the cache/MPKI
  surface. If corroborated, PHASE 3's 3-way Fulcrum should center on RSS/working-set/MPKI (the
  mandate), with the BMI2/LUT/refill graft recorded as rejected-with-mechanism (no work spent).

### MILESTONE RESIDUAL — DONE [committed 6388e0b] seam stop-point reconciliation assertion
- native_fold_parity now asserts the consumer-level STOP-POINT seam reconciliation IN-FILE (the
  advisor residual). Decodes a SECOND folded chunk at the first chunk's final_bit (windowed by the
  first's resolved tail) and asserts byte-continuity from the seam offset. 32 seams checked, all
  byte-continuous; 32/32 chunks correct, 12 flipped. Test-only (not(isal_clean_tail)); native sha
  028bd002...cb410f unchanged. A future graft now can't regress the seam silently.

### STRETCH STATUS / NEXT (pending advisor corroboration of the BMI2 rejection)
- DONE: falsifier pre-registered; ceiling BOUNDED by byte-exact A/B (NOT extrapolated) — BMI2 lever
  (a) rejected w/ mechanism (TIE; already-on; memory-bound wall); (b)/(c) already present +
  same-mechanism-covered. Guest routing assert fixed for the rename. Milestone seam residual closed.
- PENDING SUPERVISOR ADVISOR: corroborate (1) BMI2-already-on, (2) median-TIE verdict, (3) redirect
  to cache/MPKI. The graft step (charter item 2) is ANSWERED by the ceiling-bound: there is NO
  per-symbol-ALU wall to graft into on the perf build; spending the stretch on BMI2/LUT/refill would
  violate "bound the ceiling before committing" — the bound says no-op.
- PROPOSED NEXT (if corroborated): pivot to the cache mandate the wall lives in — (i) re-hook the
  byte-accounting instrument to the NATIVE per-thread working set (BOOTSTRAP_BLOCK / Block.output_ring
  128KiB u16 ring; the staged_bits hook is dead post-fold), validate via GZIPPY_MEM_BALLAST_MIB
  positive control; (ii) PHASE 3 3-way locked Fulcrum reporting wall + RSS + per-thread working-set +
  L2/L3 MPKI to locate rapidgzip's real advantage (window-apply / ring layout / copy = cache surface,
  NOT inner ALU); (iii) gzippy-isal Design-A tail before PHASE 3 per the charter sequence.

## NEW CHARTER (2026-06-06, supervisor) — TIER-APPROACH: 1.0x TIE bar, DESIGN→PROVE→ALIGN
plans/tier-approach-mandate.md SUPERSEDES "faithful but accepted-slow". Pure-Rust + inline ASM allowed.
Method is TIERED (no lever hill-climb). Leader re-driving. leader-lock held by persistent sentinel.
SUBAGENT SPAWN WORKS this session: `claude -p --model opus --permission-mode bypassPermissions "<prompt>"`.

### TIER-1 DIAGNOSIS DONE (2026-06-06) — 2 read-only research subagents + leader first-hand vendor read
THE ROOT CAUSE, source-cited (this CORRECTS the prior "memory-bound / cache-mandate" framing):
- **rapidgzip's measured ~0.46s wall is ~99% ISA-L (igzip C/SIMD).** Vendor GzipChunk.hpp dispatch:
  known-window chunk -> 100% IsalInflateWrapper (:440-444, d_c); window-absent -> pure deflate::Block
  ONLY for the <=32KiB markered prefix, then `cleanDataCount>=MAX_WINDOW_SIZE` hands the multi-MiB
  remainder to ISA-L (:520-526, d_w). The pure marker engine (the thing gzippy ports for the WHOLE
  chunk) runs <=32KiB/chunk in rapidgzip. The in-place u16->u8 flip exists in deflate::Block
  (:1282-1289) but is VESTIGIAL in production (ISA-L takes over right after it would fire).
- **u16-vs-u8 buffer-width hypothesis REFUTED.** rapidgzip uses the IDENTICAL 128KiB u16 ring
  (deflate.hpp:805 `std::array<uint16_t,2*MAX_WINDOW_SIZE>`, alignas(64), reinterpret-cast to u8 in
  place). gzippy's output_ring is a faithful port. Not narrower.
- **BMI2/per-symbol-ALU REFUTED (prior arc) AND now explained:** the gap isn't one extract op; it's
  that gzippy runs a SCALAR Rust marker loop for ~100% of bytes where rapidgzip runs igzip AVX2 ASM
  (igzip_decode_block_stateless_04.asm via multibinary dispatch) for ~99%.
- **VENDOR BENCH TABLE (deflate.hpp:72-93,137-145) is the TIER-2 ceiling evidence.** SINGLE-THREAD
  silesia (memory-bw-reduced): ISA-L 720 MB/s; rapidgzip's OWN best PURE decoder ShortBitsMultiCached-11
  337 MB/s; DoubleLiteralCached (what gzippy ported, class-of) 252 MB/s. => ISA-L is 2.1x faster than
  the best pure decoder single-thread; a pure SCALAR loop tops ~337 MB/s. Multi-thread the gap shrinks
  to 1.28x (5024 vs 3927) as the workload becomes DRAM-bandwidth-bound. gzippy measures 1.85x at T8 —
  WORSE than rapidgzip's own 1.28x pure/ISAL, i.e. gzippy's pure engine underperforms even rapidgzip's
  pure decoder, so there is BOTH pure-Rust headroom AND an irreducible ASM-class gap.
- **gzippy native per-thread DECODE hot state ~279KiB** (subagent A, cited): output_ring 128KiB +
  INLINE dist code_cache 128KiB (HuffmanCodingReversedBitsCached<30>, faithful to vendor deflate.hpp:336)
  + lit/len short LUT 16KiB. Overflows 256KiB L2, fits >=512KiB L2. Cache-hostile: 128KiB dist cache
  (scattered per-backref), backref source reads up to 64KiB back in ring. Per-chunk ~20MiB u16 buffer +
  resolve pass are DRAM-bandwidth, not residency.
- **3 PRIOR-NOTE DISCREPANCIES (subagent A, must reconcile):** (1) FIXED_TABLES/LitLenTable/DistTable
  (libdeflate path) are NOT live on native — the prior "shared tables already done" checked a DEAD path;
  live tables are per-thread, per-block-rebuilt, NO shared read-only table on the hot loop. (2) the dist
  cache is a 128KiB INLINE (unboxed) array => thread_local Block is ~279KiB, TWO 128KiB structures not
  one. (3) resolve streams the ~20MiB chunk buffer, not the 128KiB ring.
- **STALE COMMENT found (structure-mandate target):** guest_fulcrum_capture.sh:69-71 claims
  GZIPPY_BUILD_FEATURES=isal-compression gives "the SAME ISA-L clean decode rapidgzip uses (apples-to-
  apples engine A/B)". build.rs:90-96 proves FALSE post-fold: isal-compression no longer enables ISA-L
  DECODE; it would just rebuild the pure decoder. The only real-ISA-L gzippy build is gzippy-isal
  (Design-A tail, deferred). => no in-one-binary engine-swap A/B currently exists.

### CONSEQUENCE FOR THE DESIGN (TIER-1 conclusion)
The 1.0x TIE bar => gzippy-native needs a pure-Rust+inline-ASM bulk DEFLATE decoder competitive with
igzip's AVX2 inner loop for the CLEAN/known-window path (where rapidgzip spends ~99% of decode). This is
NOT the cache-mandate (buffers are faithful/same-size) and NOT BMI2 (already on, scalar-loop-bound). The
faithful-rapidgzip structure is NOT one-engine — it hands the clean tail to igzip; the governing
"one MarkerRing engine, no 2nd engine, no FFI" memory is in TENSION with a TIE (flagged to supervisor).
TIER-2 must PROVE feasibility (a standalone toy igzip-class Rust+ASM decoder benchmarked in isolation vs
ISA-L on the guest, + a bandwidth-vs-compute model from the vendor bench constants) BEFORE TIER-3 build.
Full TIER-1 design + TIER-2 proof plan: plans/tier1-design-and-tier2-proof.md (THIS checkpoint's deliverable).

### INDEPENDENT ADVISOR DISPROOF PASS [DONE 2026-06-06] — verdict: sound-with-corrections
Spawned an independent Opus advisor (read-only, disproof-driven) to attack the diagnosis by re-reading
cited source. It first-hand re-verified CLAIM 1 (ISA-L dominance, all 3 handoffs GzipChunk.hpp:440/501/520),
the 128KiB u16 ring identity, the 128KiB INLINE dist cache (huffman_reversed_bits_cached.rs 1<<15 x 4B,
NOT boxed) + no-shared-table, the BMI2 rejection, AND that the guest trace binary is the ISA-L build
(leader also confirmed: CMakeCache LIBRAPIDARCHIVE_WITH_ISAL:BOOL=ON, 42 isal symbols in nm). 5 corrections,
ALL incorporated into tier1-design-and-tier2-proof.md:
1. SEPARATE ring-STORAGE width [refuted, both 128KiB u16] from clean-bulk WRITE+RESOLVE traffic width
   [LIVE lever: gzippy writes u16 per literal marker_inflate.rs:1526 + re-streams ~20MiB u16 resolve on
   ~100% of bytes; rapidgzip ISA-L writes u8 direct isal.hpp:257 + markers only in <=32KiB prefix => ~0%].
2. High-T convergence is partly CACHE CONTENTION (vendor deflate.hpp:170-172, "LUT too large when 2 HW
   threads share a core"), NOT pure DRAM bandwidth => class-T traffic/residency levers must NOT be
   pre-ranked below class-C SIMD compute. Lever ranking was a soft tier-discipline violation; removed.
3. PROOF-2 validate-by-reproducing-the-2-walls is overfit-circular => added HOLD-OUT cross-validation
   (fit T1/T2/T4, predict T8/T16) + DIRECT perf-stat binding-term measurement as the verdict.
4. Vendor MB/s (337/720) are a Frankensystem CPU (deflate.hpp:96-107, +-20% variance) — illegitimate as
   guest targets; gate on guest-measured RATIOS (in-bench ISA-L oracle) only.
5. Added PROOF-0 PRE-GATE: cheapest decisive experiments FIRST — PG-A clean-loop slow-injection (causal
   perturbation: monotonic T8-wall response => compute binds => build SIMD; flat => SIMD moot, falsified
   in ~1hr not after a multi-week build) + PG-B u8-write+drop-clean-resolve A/B (isolates class-T traffic).
   This is the gate the BMI2 arc lacked.
Advisor bottom line: diagnosis sound to design on; design was partially mis-aimed (under-weighted class-T)
— now fixed; PROOF-1 sound, PROOF-2 fixed, PRE-GATE is the missing decisive gate, run it first.

## CHECKPOINT REACHED [2026-06-06] — TIER-1 design + TIER-2 proof plan delivered for supervisor/advisor
NO TIER-3 implementation started (charter: deliver design+proof BEFORE building). Awaiting supervisor +
independent-advisor ratification of: (1) governing-tension resolution (one no-FFI engine, igzip-class
clean inner loop via pure-Rust+inline-ASM, inner-loop divergence accepted while architecture stays
faithful); (2) the diagnosis; (3) the revised TIER-2 plan (PRE-GATE -> PROOF-1 -> PROOF-2). leader-lock
held by persistent sentinel pid 156. Subagent spawn confirmed working: claude -p --model opus
--permission-mode bypassPermissions.

---
## SUPERVISOR NOTE 2026-06-06 23:24 (lock hygiene)
First leader (TIER-1) completed and exited but left an ORPHANED `sleep 86400`
leader-lock sentinel (pid 156, ppid=1) + a redundant TIER-1 disproof advisor
(pid 4391). The orphaned sleep held the leader-lock "alive" for 24h and would
wedge any new leader's mkdir-mutex acquire. Supervisor killed both and cleared
/tmp/gzippy-leader.lock.d. TIER-2 leader (current) is the SOLE live leader.
RECOMMENDATION for any future leader: do NOT hold the leader-lock with a detached
`sleep` sentinel that outlives you — it orphans on exit. The supervisor already
enforces single-leader via pgrep before each spawn; the sleep-sentinel lock is
net-negative. Prefer no sentinel (supervisor-enforced singleton) or a lock whose
holder is the leader's own pid (auto-released on death).

---
## TIER-2 PRE-GATE — INSTRUMENT FIXED + SWEEP DONE [2026-06-07, TIER-2 leader]

### TASK 1 (instrument-validity) — DONE. Injection moved to the NATIVE clean arm.
The prior subagent had the slow-injection in resumable.rs (Engine C / gzippy-ISAL
path) with a comment claiming "FlipToClean tail runs 100% through resumable" —
WRONG for native. On gzippy-native (the 1.0x bar) the fold keeps Engine M
(marker_inflate::Block) decoding the clean tail in-place (gzip_chunk.rs:1219,
not(isal_clean_tail)); ~99% clean bytes decode through marker_inflate's
CONTAINS_MARKERS=false arm. Injection is now wired into BOTH native clean sites:
read_internal_compressed_specialized::<false> (the lut_litlen multi-symbol loop,
the live native dynamic-Huffman path) AND read_internal_compressed_canonical_
specialized::<false>. resumable injection KEPT (correct site for the isal control).
PROVEN on perf-target build (Rosetta x86_64 gzippy-native, path=ParallelSM,
silesia): GZIPPY_SLOW_HITS counter = 40,131,993 clean decode events for the ~203MB
decode (∝ clean bytes, NOT ~0). Site is the live native clean loop. Commit d0aa1db.

### TASK 2 (self-test) — PASS.
- OFF byte-exact 028bd002...cb410f on BOTH gzippy-native AND gzippy-isal (x86_64),
  T1 + T8, path=ParallelSM. OFF == identity by construction (hoistable spin==0).
- POSITIVE CONTROL (locked harness, silesia-large 503MB, N=9): T1 F=0 3.7340s ->
  T1 F=100 spin 6.3833s = +71% (sd 0.1%, massively out of spread). Knob really
  slows the loop. The pause-hint "sleep" was too cheap to be a valid freq-neutral
  control (~7% at 5x), so the sleep kind was reimplemented as a REAL batched
  thread::sleep (yields the core, calibrated to the spin per-iter cost).

### TASK 4 — PRE-GATE SWEEP (locked guest harness, silesia-large 503MB, T8, N=9 interleaved, sha-verified e114dd2b..., diverged=0 every run, RUN_TRUSTWORTHY=true)
| F   | SPIN T8 wall | Δ vs F0 | SLEEP T8 wall | Δ vs F0 |
|-----|-------------|---------|---------------|---------|
| 0   | 1.1209s     | —       | 1.1209s       | —       |
| 25  | 1.1597s     | +3.5%   | 1.1397s       | +1.7%   |
| 50  | 1.2281s     | +9.6%   | 1.1990s       | +7.0%   |
| 100 | 1.4368s     | +28.2%  | 1.2411s       | +10.7%  |
(within-arm sd 0.8–4.4%; every Δ at F=50/100 is out of spread under BOTH kinds.)
T1 positive control: F0 3.7340s -> F100 spin 6.3833s (+71%, sd 0.1%).

### VERDICT: COMPUTE-BOUND (clean-loop compute IS on the T8 critical path).
- MONOTONIC & PROPORTIONAL T8 response under SPIN (3.5/9.6/28.2%), out of spread.
- The rise SURVIVES the frequency-neutral SLEEP control (1.7/7.0/10.7%, monotonic,
  out of spread) — so the criticality is REAL, not a turbo artifact. Per the
  pre-registered falsifier (plans/pre-gate-falsifier.md), monotonic-survives-sleep
  ⇒ COMPUTE-BOUND ⇒ proceed to PROOF-1 (the SIMD-compute proof).
- Spin slope (28%) > sleep slope (11%) at F=100: the gap IS the turbo-depression
  the busy-spin causes (the exact confound the control isolates); the sleep slope
  is the turbo-clean lower bound and it is still clearly positive.
- DIAGNOSTIC: T1 F100 +71% vs T8 F100 +28% (spin) — at T8 the clean-loop compute
  is partially overlapped by parallelism (wall is less compute-elastic than T1)
  but STILL on the critical path. NOT flat ⇒ class-C (SIMD compute) is NOT moot.
- 1b (u8-write / traffic A/B) NOT run in this window — flagged as the immediate
  next experiment to rank class-T traffic as co-lever (matrix row: 1a MONOTONIC,
  1b TBD ⇒ at minimum COMPUTE-BOUND→PROOF-1; class-T co-lever pending 1b).

NEXT: supervisor + independent disproof advisor gate this verdict BEFORE PROOF-1/
PROOF-2 or any SIMD build (per charter SEQUENCE). No TIER-3 started.

### TASK 5 — INDEPENDENT DISPROOF ADVISOR [DONE] verdict: CORROBORATE-WITH-CAVEATS
Full verdict: plans/pre-gate-advisor-verdict.md (independent Opus, read-only, read
the cited source). It could NOT construct a path to FLAT/bandwidth-bound — the
narrow claim survives. Corrections folded in (the VERDICT block above is AMENDED):
- RELABEL: drop "COMPUTE-BOUND" (overreach). Correct claim = "clean-loop compute
  is ON the T8 critical path; class-C (SIMD compute) is NOT moot." The wall is
  MIXED; clean compute is only ~11–29% of the T8 wall (sleep floor 11% vs spin/T1
  ~29% — methods in genuine tension; the T1 positive control +71% extrapolates to
  ~29%, matching SPIN, so the usual "trust the sleep floor" likely UNDER-reports).
- RULE 3 (mandatory): slowing the loop adds wall ≠ speeding it pays. The pre-gate
  licenses ONLY "not-moot." PROOF-1 = the REMOVAL ORACLE (bound the speed-up
  ceiling) is MANDATORY before ANY SIMD build. Do NOT start the SIMD stretch on
  this pre-gate alone.
- BIGGER LEVER (advisor): even infinite clean speed-up -> 0.79–1.00s vs rapidgzip
  0.53s — still ~1.5–2x slower. T8 runs ~42% parallel-efficiency (1.121s vs 0.47s
  ideal); ~58% lost to placement / head-of-line stalls (confirmed-offset-prefetch-
  gap, ~40% of T8). THAT outranks class-C for closing the rapidgzip gap. Weigh the
  placement lever before committing to the SIMD project.
- INSTRUMENT: site re-asserted on the EXACT 503MB guest T8 build — GZIPPY_SLOW_HITS
  = 94,887,526 clean decode events via path=ParallelSM (∝ clean bytes). Treat
  F->%wall as approximate (T1-calibrated); good for monotonicity, not magnitude.
- 1b (u8-write/traffic A/B) still not run — the next experiment to rank class-T.

---
## TIER-2 STEP A — REMOVAL ORACLES [2026-06-07, STEP-A leader, fresh instance]
CHARTER: plans/tier2-revised-placement-primary.md. Bound BOTH ceilings with removal
oracles (CLAUDE.md rule 3) BEFORE any design commit. Guest VERIFIED IDLE (load 1.04,
no stray gzippy/cargo/rapidgzip/harness procs; /dev/shm clean). HEAD ce9fe6f.

### Falsifiers PRE-REGISTERED (before any run)
- plans/step-a-oracle-c-falsifier.md (clean-compute removal). Mechanism = existing
  decode-bypass FLOOR (PASS A) with the prior leverB CONTAMINATION fixed (prebuilt-map
  init out-of-wall; /dev/shm capture; swap gate; region-removed proof = clean-decode
  span→≈0 in floor trace). Prediction to beat: ~0.79-1.00s (advisor) / 0.78-0.84 (model).
- plans/step-a-oracle-p-falsifier.md (perfect-placement removal — DECISIVE). Mechanism
  P1 = trace-derived optimal-makespan (LPT) bound on the EXISTING locked-Fulcrum T8
  per-chunk decode-busy times (placement-free lower bound, no new code) + self-tests
  (sum-conservation, max-chunk, Σ/8); P2 = bypass-FLOOR placement-residual cross-check.
  Verdict: FLOOR_P ≤ ~0.55s ⇒ placement SUFFICIENT (proven path to tie).

### KEY INFRA FINDING (saves a build): the Oracle-C engine ALREADY EXISTS and is byte-exact
decode_bypass.rs CAPTURE/REPLAY (GZIPPY_BYPASS_CAPTURE/DECODE) + guest_ceiling.sh PASS A.
Prebuilt path (decode_bypass.rs:460-471) = HashMap lookup + Option::take, no per-call
memcpy. Prior leverB run (plans/leverB-ceiling.md) read 3.667s = LOAD-CONTAMINATED
(one-time prebuilt rebuild in-wall + swap), declared a HYPOTHESIS not a ceiling, owed a
re-run "from RAM, prebuilt out-of-wall". THIS is that re-run.

### ORACLE-P [DONE — trace-derived, validated] → PLACEMENT SUFFICIENT
plans/step-a-oracle-p-falsifier.md MEASURED RESULTS. Clean T8 trace (000148, HEAD
d0aa1db = decode-identical parent of ce9fe6f). FLOOR_P (placement-free single-pass
LPT makespan, gzippy engine fixed) = **0.41-0.49s ≤ rapidgzip 0.524s** ⇒ perfect
placement reaches the tie band WITHOUT touching the engine. Positive control:
actual/makespan = 1.36 on BOTH tools. MECHANISM: 5.90s of gzippy's 6.33s T8 decode
busy = worker.scan_candidate (speculative re-decode at mispredicted boundaries);
clean decode (isal_stream_inflate) only 0.365s; single-pass T1 = 3.734s. rapidgzip
decode busy 2.99s (confirms boundaries ahead, never re-scans). VERDICT: PLACEMENT is
the proven primary lever (faithful rapidgzip boundary-confirmation port). CAVEAT
(honest): bound assumes the redundant scan is eliminable + single-pass balances — the
port-feasibility question, NOT proven by the ceiling (advisor to attack).

### ORACLE-C [RE-RUNNING] — first attempt ORPHANED (lesson)
SUBAGENT-ORPHAN LESSON: the `claude -p` Oracle-C subagent printed an interim message
and EXITED (treated the backgrounded guest run as auto-reinvokable) → SIGHUP killed
its child driving-ssh → host_lock died → watchdog restored freq state → run produced
only the build, no captures. FIX: leader runs the guest harness DIRECTLY via a
background Bash task (parented to the persistent runner, not an exiting subagent).
Re-run launched (task b486p1g4b, /tmp/oracle-c-run2.log) at HEAD a49c357 (warm_prebuilt
fix + routing-assert fix both in). Guest verified idle (load 1.08, shm clean) before launch.

### ORACLE-C [DONE — run2 a49c357] → class-C floor ~0.4-0.7s (instrument-entangled), GREY
plans/step-a-oracle-c-falsifier.md MEASURED RESULTS. Locked harness, RESTORE VERIFIED.
HARD gates PASS (sha=OK byte-exact, hit%=97.6, sd%=0.9, anchor 1.13s reproduced, no
swap-thrash). REGION-REMOVED PROOF passes (decode_chunk busy 6.33s→0.076s in floor
trace). BUT the raw CEIL_FLOOR_A=3.63s is DOUBLY contaminated: (1) +3.1s warm_prebuilt
still in the whole-process wall (my fix moved it before drive_t0 but the harness times
the process, not drive_t0 — fix incomplete); (2) decode≈0 frees windows ⇒ L_resolve
collapses (162µs vs 19.93ms) ⇒ fulcrum critpath 336ms UNDER-states. Robust bracket:
A2 0.444 / critpath 0.336 / A−warm−load ~0.5 / B-sleep66 0.732 ⇒ class-C floor ~0.4-0.7s,
BELOW the advisor's 0.79-1.00s. KEY: full decode-removal is a DEGENERATE oracle — it
co-collapses the publish-chain it should preserve. ⇒ class-C bounded (≤~0.7s floor ⇒
infinite clean speedup buys ≤~0.4s/37% of wall), NOT the clean lever Oracle-P is.

### BOTH CEILINGS LANDED — CHECKPOINT
- Oracle-P (placement-free): **0.41-0.49s** ≤ rapidgzip 0.524s ⇒ PLACEMENT SUFFICIENT.
- Oracle-C (decode-free): **~0.4-0.7s** (entangled, GREY) ⇒ class-C bounded-secondary.
- Both land in the same ~0.4-0.7s band because gzippy's two costs (redundant
  speculation = placement; publish-chain) OVERLAP in the parallel pipeline.
- IMPLICATION: placement is the proven primary path to the 1.0× tie (faithful rapidgzip
  boundary-confirmation-ahead port); class-C/class-T secondary. The 5.9s scan_candidate
  (speculative re-decode) is the single largest recoverable item.
NEXT: spawn independent disproof advisor (read-only) → plans/step-a-oracle-advisor-verdict.md.
THEN STOP for supervisor gate. Do NOT start STEP B/C/D or TIER-3.

### DISPROOF ADVISOR [DONE] → CHECKPOINT (STOP for supervisor)
plans/step-a-oracle-advisor-verdict.md. Read-only, re-derived ALL numbers first-hand
(all reproduce). VERDICT: Oracle-P "PLACEMENT SUFFICIENT" REFUTED → NECESSARY-BUT-
INSUFFICIENT; Oracle-C CORROBORATE-WITH-CAVEATS (degenerate, GREY). Key correction:
ramp-consistent placement-perfect gzippy = 0.56-0.66s vs rapidgzip 0.524s (7-26% loss);
the 5.9s scan_candidate is first-pass MARKER decode (decoded once, expensively), not
redundant re-decode; an ENGINE residual survives perfect placement (gzippy clean
91ms/chunk vs rapidgzip 39ms = 2.3×) ⇒ class-C is CO-PRIMARY, not bounded-secondary.
CHECKPOINT REACHED. Did NOT start STEP B/C/D or TIER-3. Awaiting supervisor gate.
OWED before STEP-C (advisor): a CLEAN-ONLY T8 removal oracle (force all chunks through
isal_stream_inflate w/ predecessor windows, measure busy) = the least-entangled engine
ceiling signal both oracles missed.

## TIER-2 STEP A.2 — CLEAN-ONLY ENGINE ORACLE [DONE 2026-06-07, A.2 leader, fresh instance]
CHARTER plans/step-a2-clean-only-oracle.md. Falsifier PRE-REGISTERED before any run:
plans/step-a2-clean-only-falsifier.md. HEAD e89006b0 (commits: instrument 4bd8ecb7,
guest harness 96f98a16, driver e89006b0). Guest verified idle before launch; host
RESTORE VERIFIED after (no_turbo=0, all 8 guests thawed, /dev/shm seed file removed).

### INSTRUMENT (byte-exact, env-gated) — NOT the broken/degenerate prior oracles
- NEW src/decompress/parallel/seed_windows.rs. GZIPPY_SEED_WINDOWS_CAPTURE (p=1 records
  aligned start_bit→window pairs at the NATURAL clean path) + GZIPPY_SEED_WINDOWS
  (pre-seed block_finder w/ aligned boundaries + clean-path window fallback). FORCES
  every chunk clean while KEEPING real Huffman decode AND the production consumer/publish
  chain — distinct from (a) the BROKEN drive_clean_window_oracle (GZIPPY_CLEAN_WINDOW_ORACLE,
  bypasses the whole scheduler+consumer with a bespoke loop) and (b) the DEGENERATE
  Oracle-C/decode_bypass (decode≈0 → publish-chain collapse).
- STRUCTURAL FINDING (proven while building): clean-only is INCOMPATIBLE with production
  speculative dispatch UNLESS boundaries are confirmed ahead — clean decode is only
  possible at a real deflate boundary, but dispatch uses spacing GUESSES whose
  WindowMap-key windows are MISALIGNED to the guess (first naive WindowMap-snapshot capture
  hit 0%/diverged). Engine-isolation REQUIRES confirmed-boundary dispatch = the placement
  fix itself ⇒ placement & engine are structurally COUPLED.

### SELF-TEST PASSES (CLAUDE.md rule 4) — proves it isolated engine + preserved publish chain
silesia-large 503MB, T8 (guest 199, locked, sha e114dd2b…):
- forced-clean: SEED window_seeded 4→39 (ALL clean), finished_no_flip 38→0 (zero marker
  decode), fused_lut 35→0 (zero marker resolution); hit%=100.0, sha=OK byte-exact.
- publish-chain-preserved: Early window publish=39 in BOTH normal AND seeded (NOT collapsed
  like Oracle-C); seeded trace consumer.window_publish_clean = 2.2ms total / 0.06ms per chunk
  (running, just cheap with no markers).
- off==identity (env unset = today's decode, by construction); routing 43/0 lib tests.

### RESULT — ENGINE CEILING (the deliverable). VERDICT: ENGINE-IS-RESIDUAL (co-primary CONFIRMED)
| arm (T8, N=9 interleaved, sha-OK) | min | median | sd% | MB/s |
|---|---|---|---|---|
| gzippy clean-only (ENGINE ceiling, publish chain intact) | 0.5896 | **0.6134** | 1.9 | 854 |
| rapidgzip | 0.5347 | 0.5396 | 3.3 | 942 |
| gzippy normal (baseline) | 1.1157 | 1.1244 | 1.1 | 451 |
- **clean-only 0.6134 vs rapidgzip 0.5396 = +13.7% (Δ0.0738s ≫ ~2-3% spread) → NOT a TIE.**
- Lands in the PRE-REGISTERED [0.58,0.72] ENGINE-IS-RESIDUAL band. Engine gap SURVIVES
  forcing every chunk clean ⇒ the residual is the ENGINE, confirming class-C is co-primary.
- Per-chunk CLEAN busy (seeded trace, 39 chunks): worker.isal_stream_inflate = **92.7ms/chunk**
  — corroborates the advisor's independent 91ms derivation; vs rapidgzip 39ms = **2.38× engine
  gap**, ramp-consistent. Implied engine-bounded wall = 39×92.7/8×1.36 ≈ 0.61s == measured.
- RAMP-CONSISTENT (the STEP-A error fixed): clean-only WALL (0.61) vs rapidgzip WALL (0.54),
  same treatment — no floor-vs-wall mismatch. Decode busy 3616ms/8 = 0.452 ideal, ×1.36 = 0.61.

### CHECKPOINT — engine ceiling cleanly bounded. The grey 0.4-0.7s (Oracle-C) RESOLVED to ~0.61s.
- Placement ceiling (Oracle-P, ramp-consistent): ~0.56-0.66s. Engine ceiling (this, A.2): ~0.61s.
  rapidgzip 0.54s. BOTH levers needed for the 1.0× tie; both are real, both ~13% gaps.
- 1b (u8-write traffic A/B) NOT run this window (engine oracle was the priority + landed; flag
  as still-owed to rank class-T). Did NOT start STEP-C design revision or any TIER-3.
NEXT: spawn ONE independent disproof advisor (read-only) → plans/step-a2-advisor-verdict.md to
attack the oracle's validity (engine isolated? publish-chain preserved? ramp consistent?), then
STOP for supervisor gate.

## TIER-2 STEP C — REVISED DESIGN [DONE 2026-06-07, STEP-C leader, fresh instance]
CHARTER plans/step-c-revised-design.md. TIER-2 PROVE complete; this synthesizes the proven
ceilings into the revised TIER-1 design (the TIER-3 gate). HEAD b8a38e64. No guest measurement
run this turn (design synthesis + read-only subagents only); no host state touched.

### DELIVERABLE — plans/tier1-design-v2.md (the revised single coherent architecture)
- §1 PLACEMENT (faithful port): read-only subagent mapped rapidgzip's 5 scheduler mechanisms
  to gzippy two-column, source-cited (/tmp/scheduler-vendor-map.md). FINDING: 3 already FAITHFUL
  (prefetch strategy, window propagation, consumer-overlap join), 1 PORTED-BUT-DEFEATED
  (confirmed-boundary dispatch), 1 MISSING = the prime defect: INTERIOR/SUBCHUNK REUSE
  (getIndexedChunk, vendor GzipChunkFetcher.hpp:254-309). gzippy BUILDS unsplit_blocks
  (chunk_fetcher.rs:2752) + block_map.find_data_offset (block_map.rs:135) but NEVER READS them.
  Corrects a STALE memory note: matches_encoded_offset IS a range check (chunk_data.rs:461-469),
  not == equality — the blocker is missing interior-EMIT, not the accept predicate.
- §2 ENGINE (inner-loop open territory): E1 u8-direct clean write (the 1b sub-lever, low-risk
  first), E2 wide SIMD back-ref copy, E3 packed multi-literal store (ca52389 non-binding), E4
  wide refill — each isolation-benchmarked vs an in-bench ISA-L positive control (guest ratio,
  NOT Frankensystem absolutes) BEFORE integration. Governing-tension resolution STANDS: one
  no-FFI engine, inner loop reimplemented igzip-class; inner-loop divergence accepted, arch
  faithful (needs supervisor ratification).
- §3 REACHABILITY (coupled, not additive): placement-alone ⇒ ~0.61s (A.2 MEASURED, not rescaled);
  placement+engine-at-igzip-class ⇒ decode stops binding, wall ⇒ shared pipeline floor.
- §4 STRUCTURE mandate sequenced; §5 TIER-3 sequence; §6 checkpoint.

### INDEPENDENT DISPROOF ADVISOR [DONE] → plans/step-c-design-advisor-verdict.md
VERDICT: ratify the METHOD, not the CONCLUSION. All 3 targets CORROBORATE-WITH-CAVEATS:
- §3: coupled-not-additive is sound + double-count-free; 0.61s is measured. BUT the 0.61→0.54
  "TIE" step assumes gzippy's NON-decode floor = rg 0.54s — contradicted by a MEASURED ~225ms
  consumer-serial term + 0.497s consumer block (neither shrinks with engine speed). Read §3 as
  ≥0.54s, plausibly 0.54-0.62s, TIE only at the optimistic edge.
- §1: range-check correction right + Subchunk.decoded_offset exists; BUT "just wire dead code"
  too glib — find_data_offset is DECODED-BYTE-keyed for a consumer gzippy doesn't have (gzippy
  is encoded-BIT-keyed), unsplit_blocks only built for subchunks>1, AND the parent-cached
  precondition was MEASURED TO FAIL (318ms lag → eviction) + left as an UNANSWERED discriminator.
- §2: E1/E3 genuinely unbuilt (re-attempt justified); guest-ratio ISA-L control is a sound ENGINE
  gate; BUT 2.1× pure-decoder ceiling is a hard headwind + the bench gates engine, not wall.
- BOTTOM LINE: safe to authorize TIER-3 as a MEASUREMENT-GATED INVESTIGATION; do NOT ratify the
  headline "1.0× tie reachable." ONE thing most likely to break it: gzippy's non-decode
  consumer-serial floor is structurally higher than rg's. Add a 3rd pre-registered measurement
  (decompose the 0.61s consumer block into decode-wait vs serial) BEFORE the engine build.

### CAVEATS FOLDED BACK INTO plans/tier1-design-v2.md (design ⇄ verdict now consistent)
- §1.2: added THREE port pre-conditions (coordinate-system mismatch; single-subchunk map gap;
  parent-cached discriminator — answer FIRST). The faithful port is getIndexedChunk's LOGIC in
  gzippy's encoded-bit offset space, NOT changing the consumer request model (that = forbidden
  redesign).
- §3: added the NON-DECODE FLOOR caveat (≥0.54s) + the OWED THIRD MEASUREMENT (consumer-block
  decompose). Verdict now: tie reachable CONDITIONAL on TWO floors (engine ≤39-45ms AND
  non-decode floor ≤0.54s); else add a 4th off-critical-path consumer lever or revisit the bar.
- §5: added SEQUENCE step 0 = the two cheap byte-exact discriminators (parent-cached probe +
  consumer-block decompose) run FIRST, before any port/build.

### CHECKPOINT REACHED — STOP for supervisor ratification (TIER-3 gate). NO TIER-3 started.
Subagents (vendor map + advisor) were READ-ONLY, synchronous, collected in-turn; both process
trees explicitly killed (no orphans). No guest run, no host state change, no build this turn.
Supervisor to ratify: (1) the method + sequence; (2) the governing-tension resolution (§2.1);
(3) that TIER-3 is authorized as a measurement-gated investigation (not a ratified tie claim),
with step-0 discriminators run before the placement port and before the engine build.

## TIER-3 STEP 0 — TWO DISCRIMINATORS [IN PROGRESS 2026-06-07, STEP-0 leader, fresh instance]
CHARTER plans/tier3-step0-authorization.md. STEP 0 ONLY (two cheap discriminators); NO placement
port, NO engine work. HEAD b8a38e64. Falsifiers PRE-REGISTERED BEFORE any run:
plans/step0-discriminator-a-falsifier.md (parent-cached) + plans/step0-discriminator-b-falsifier.md
(consumer-block decompose + HARD ESCALATION GATE >0.54s).
Guest 199 verified idle (load 0.51 falling, no stray procs, /dev/shm only stale debug files,
no_turbo=0 = unlocked). Leader-lock: none held (supervisor enforces singleton; NO sleep sentinel
per the 2026-06-06 23:24 lock-hygiene note). Disk 30Gi free (94%) — watching df around builds.

### DISCRIMINATOR (b) — CONSUMER-BLOCK DECOMPOSE [PRELIMINARY from existing A.2 trace] → PASS (floor ≈18ms ≪ 0.54s)
Instrument: scripts/bench/consumer_block_decompose.py — per-span SELF-time on the consumer tid
(tid=1) with no-double-count (nested children subtracted, CLAUDE.md rule 8). WAIT vs SERIAL
classification done by READING THE SOURCE (not guessing): blocking-wait spans are
wait.block_fetcher_get / wait.future_recv / ttp.rx_recv_block (block_fetcher.rs:247 rx.recv_timeout
pump) / consumer.dispatch_recv (chunk_fetcher.rs:3051 rx.recv()); everything else = serial CPU.
- PHANTOM-LEVER CAUGHT: a first (wrong) classification with only {block_fetcher_get, future_recv}
  in the wait set put NORMAL-path SERIAL at 0.515s — but ttp.rx_recv_block (0.26s) + dispatch_recv
  (0.19s) are BLOCKING WAITS, not CPU. The advisor's "~225ms consumer-serial" is this same
  misattribution: those spans are decode-wait, not bookkeeping. Source-read dissolved it.
- Operating point = the A.2 CLEAN-ONLY oracle trace (trace_seed_T8.json, GZIPPY_SEED_WINDOWS,
  every chunk clean at a confirmed boundary = the placement-PERFECT operating point §3 prices the
  0.61s consumer block at). Conservation PASSES (sum-self vs wall gap 0.46%).
  | trace | consumer_wall | DECODE-WAIT | SERIAL-BOOKKEEPING (the floor) |
  |---|---|---|---|
  | clean-only (placement-perfect) | 0.506s | 0.489s (96.5%) | **0.0178s (3.5%)** |
  | normal (marker baseline)       | 1.048s | 0.979s (93.4%) | 0.069s (6.6%) |
- VERDICT (preliminary): non-decode SERIAL floor ≈ 0.018s ≪ 0.54s ⇒ HARD GATE PASSES with a
  huge margin; the consumer does NOT structurally forbid the tie. STILL OWED before final:
  the SLOW-knob positive control (decode slow-injection must inflate DECODE-WAIT while SERIAL
  stays flat — pre-registered) + N≥7 robustness, from a fresh locked guest run.

### DISCRIMINATOR (a) — PARENT-CACHED-AT-STALL [DONE — locked guest] → NO (NOT_RESIDENT) ⇒ RE-SCOPE placement
Probe GZIPPY_STALL_RESIDENCY_PROBE (commit 76e9dc9d, stall_residency.rs) at the cold-get None
branch. Locked guest 199, silesia-large 503MB, T8, host-lock GATE PASS, RESTORE VERIFIED.
Byte-exact every run (sha=OK == ref e114dd2b…); OFF==identity match=YES.
- VERDICT RUN (default caps): total=4 startup=1 CONTAINING_CACHED=0 CONTAINING_IN_FLIGHT=0
  NOT_RESIDENT=4 → 0% of non-startup stalls have a resident containing parent.
- Per-stall detail (auditable, nearest_le_start = nearest cached start ≤ decode_start): EVERY
  non-startup stall has nearest_le_start:-1 — NO cached/in-flight chunk has a start ≤ the stalled
  offset; ALL resident + in-flight chunks are AHEAD of decode_start. The consumer is BEHIND its
  prefetch frontier; the containing chunk was already consumed/passed. Genuinely EVICTED/absent.
- POSITIVE CONTROL passes (probe tracks residency): cap=1 (tiny) total 4→9 NOT_RESIDENT 4→9 (more
  eviction → more cold-get stalls); cap=256 (huge) stays total=4, still 0% resident — proving this
  is NOT a cache-CAPACITY problem (default cap 16 already holds the set); it is consumer-PACE/overshoot.
- ANSWER = NO. Per the pre-registered verdict rule, NOT_RESIDENT majority ⇒ the gap is
  cache-residency/consumer-pace, NOT interior-reuse-fixable. The getIndexedChunk/interior-EMIT port
  (tier1-design-v2 §1.2) presupposes a resident parent it does NOT have ⇒ that placement recipe must
  be RE-SCOPED: the fix is UPSTREAM of interior reuse (make the consumer keep pace / keep the
  containing chunk in-flight that rapidgzip keeps). Chicken-and-egg the advisor flagged. Confirms
  [[project_confirmed_offset_prefetch_gap]] ROOT CAUSE (consumer lags prefetcher ~318ms).

### DISCRIMINATOR (b) — CONSUMER-BLOCK DECOMPOSE [DONE — locked guest] → PASS (floor ~0.015s ≪ 0.54s)
Locked guest, clean-only oracle (GZIPPY_SEED_WINDOWS = placement-perfect operating point),
GZIPPY_TIMELINE trace_v2 spans, output→/dev/null (T-invariant sink), 3 baselines + SLOW control.
consumer_block_decompose.py self-time (no double-count); WAIT spans = block_fetcher_get/future_recv/
ttp.rx_recv_block/dispatch_recv (source-classified). sha=OK byte-exact every run.
| trace | consumer_wall | DECODE-WAIT | SERIAL-BOOKKEEPING (floor) |
|---|---|---|---|
| clean  | 0.506s | 0.493s (97.5%) | 0.0127s |
| clean2 | 0.507s | 0.492s (97.0%) | 0.0151s |
| clean3 | 0.500s | 0.482s (96.5%) | 0.0176s |
| SLOW+100% | 5.142s | 5.097s (99.1%) | 0.0448s |
- non-decode SERIAL floor = 0.013–0.018s (mean ~0.015s, spread ~0.005s) ≪ 0.54s ⇒ HARD GATE PASSES
  with ~36× margin. The consumer does NOT structurally forbid the tie.
- POSITIVE CONTROL passes the pre-registered falsifier: decode +100% (78.7M clean-loop inject hits
  confirmed) inflated DECODE-WAIT 0.49→5.10s while SERIAL stayed 0.013→0.045s (FLAT) ⇒ the decompose
  correctly isolates decode-wait from serial.
- CONFOUND CAUGHT + FIXED: an interim run had output→disk(mktemp), inflating consumer.writev to a
  fake 0.267s "floor". Streaming output→/dev/null removed it. The advisor's feared ~225ms
  consumer-serial term was the decode-WAIT spans (rx_recv_block/dispatch_recv) mis-bucketed as
  serial — dissolved by source-reading the span semantics.

### STEP-0 CHECKPOINT (STOP for supervisor gate)
- (a) parent-cached = NO (0% resident, 318ms-lag eviction confirmed) ⇒ tier1-design-v2 §1.2
  interior-reuse/getIndexedChunk port is NOT the fix as written; placement must be RE-SCOPED to a
  consumer-pace fix (faithful: rapidgzip's consumer stays ~0-17ms behind its prefetcher; gzippy lags
  ~318ms). NOT a "wire dead code" port.
- (b) non-decode floor = ~0.015s ≤ 0.54s ⇒ CONTINUE (no escalation; consumer does not forbid the
  tie). The ENGINE front remains gated by §2.3's isolation bench (not run this turn).
- NEXT: independent disproof advisor (read-only) → plans/step0-advisor-verdict.md, then STOP for the
  supervisor gate. NO placement port, NO engine work started. Host RESTORE VERIFIED; no orphan procs.

### INDEPENDENT DISPROOF ADVISOR [DONE] → both conclusions SAFE (CORROBORATE-WITH-CAVEATS)
plans/step0-advisor-verdict.md (independent Opus, read-only, re-derived from source).
- (a) CORROBORATE-WITH-CAVEATS. PRIME SUSPECT CONFIRMED: my original CONTAINING_CACHED counter
  was VACUOUS — [enc,max] is the speculative START-tolerance window (decoded bytes BEGIN at max,
  chunk_data.rs:143-145; re-anchored chunks have enc==max), so it could never fire for a real
  containing parent. BUT the NO verdict SURVIVES on the bug-free channels: nearest_le_start:-1 on
  every stall (correct necessary-condition test) + CONTAINING_IN_FLIGHT=0 (keyed). The port reads
  the parent from cache(); nearest_le_start:-1 proves it isn't there. Advisor's owed item ANSWERED.
- (b) CORROBORATE-WITH-CAVEATS. Honest; source dissolves the ~225ms fear (apply_window runs on the
  POOL, the consumer blocks on a future wrapped as wait.future_recv NESTED inside
  dispatch_post_process — correctly bucketed DECODE-WAIT). Robust across operating points (even
  NORMAL marker-heavy trace serial=0.069s ≪ 0.54s). CAVEAT: /dev/null excludes production output-
  write (writev ~0.245s on a real sink) — the real floor is ~0.015s + output-write; still passes,
  fair only if rapidgzip's 0.54s is same-sink. Keep the tie contingent on engine ≤39-45ms AND a
  same-sink floor ≤0.54s.

### CORRECTION RUN (commit d764734c, locked guest) — advisor fix applied, verdict HELD
Re-snapshot the encoded END (enc+encoded_size_bits); test enc<decode_start<encoded_end; added the
bug-free has_nearest_le_start channel. Byte-exact OFF==identity (028bd002…cb410f / e114dd2b…).
RESULT (all sha=OK): **has_nearest_le_start=0 at ALL cap settings — default(16), cap1, AND cap256
(huge).** Even with an effectively unbounded cache NO resident chunk starts at/below the stalled
offset ⇒ rules OUT capacity-eviction, CONFIRMS never-retained / consumer-pace (containing chunk
consumed+passed before the consumer reached it; lags the frontier ~318ms). The interior-reuse port
has no resident parent to reuse REGARDLESS of cache size. NO verdict now on bug-free, cap-swept evidence.

### STEP-0 FINAL (STOP for supervisor)
(a) parent-cached = **NO** (consumer-pace / never-retained, cap-swept, advisor-confirmed) ⇒
   tier1-design-v2 §1.2 interior-reuse/getIndexedChunk port is NOT the fix as written; placement
   RE-SCOPES to a consumer-pace fix (keep the containing chunk in-flight; rapidgzip's consumer stays
   ~0-17ms behind its prefetcher vs gzippy ~318ms). Faithful (rapidgzip does this), just a different
   mechanism than interior-EMIT.
(b) non-decode floor = **~0.015s (≪ 0.54s)** ⇒ CONTINUE; consumer does NOT structurally forbid the
   tie. Caveat: re-quote with production output-write for the true same-sink margin.
No placement port / engine work started. Both subagents (advisor) read-only, synchronous, no orphans.
Host freq RESTORE VERIFIED after every guest run. Awaiting supervisor gate.

## PLACEMENT RE-SCOPE — LAG-CAUSALITY DIAGNOSIS [DONE 2026-06-07, placement-rescope leader]
CHARTER plans/placement-rescope-diagnosis.md. Answer the pivotal question: is the ~318ms
consumer-prefetcher lag STRUCTURAL or an EFFECT of the 2.38x slow engine? HEAD cb60842d.
Falsifier PRE-REGISTERED before any run: plans/lag-causality-falsifier.md. Guest verified idle
before; host RESTORE VERIFIED after (no_turbo=0, all guests thawed); freed ~1.7G stale guest
clones (gzippy-{reimplement,consolidate}-verify) for build headroom. No orphans.

### PERTURBATION (slow_knob sweep, native clean arm marker_inflate.rs:1307/1546, locked guest T8, N=7, sha-verified, NO DIVERGENCE)
| combo | T8 wall | per-blk busy | STALL COUNT (non-startup) | wait.block_fetcher_get | wait/blk-busy |
|---|---|---|---|---|---|
| F0 | 1.1105s | 0.29963ms | 3 | 369.1ms | 1232 |
| F50 spin | 1.2055s | 0.35335ms | 3 | 445.0ms | 1259 |
| F100 spin | 1.3704s | 0.44310ms | 3 | 646.3ms | 1459 (+18.4%) |
| F50 sleep | 1.1827s | 0.32327ms | 3 | 418.5ms | 1295 |
| F100 sleep | 1.2340s | 0.36314ms | 3 | 507.5ms | 1398 (+13.5%) |
Wall monotonic + survives sleep (clean compute on T8 crit path — known). STALL COUNT dead-flat
at 3 across 0->+48% slowdown (both kinds). Injection symmetric (slows prefetch+on-demand alike).

### VERDICT (advisor-corrected): MIXED — existence STRUCTURAL, magnitude materially ENGINE-COUPLED, separability UNPROVEN
plans/lag-causality-verdict.md. First-draft "STRUCTURAL/engine-invariant, small residual" was
OVERSTATED + the design rested on a VENDOR MISREAD; both corrected after the disproof advisor.
- EXISTENCE structural: load-bearing signal has_nearest_le_start=0 (NOT the flat count — count is
  a saturated/low-N/wrong-direction proxy per rule 3). The 3 overshoot-tail cold-gets exist at F0,
  cost ~369ms cold-wait now, parent never-retained. Removing them is worth doing.
- MAGNITUDE engine-coupled: normalized wait/blk-busy +18.4% spin / +13.5% sleep (survives the
  frequency control) => ~24% of cold-wait growth under slowdown is genuine drift. The COST (wall)
  is partly engine-set; a faster engine shrinks the 318ms cost while the count stays 3.
- SEPARABILITY unproven: placement does NOT dissolve into engine (stalls cost wall at F0) but is
  NOT cleanly separable co-primary either. Both levers stay co-primary; engine still +13.7% (A.2).

### VENDOR PACING MAP [read-only subagent, plans/vendor-pacing-map.md]
rapidgzip's ~0-17ms pacing STRUCTURE (prefetch depth 2xpar, separate un-evictable in-flight map,
join-in-flight, lean off-path consumer post-proc) is FAITHFULLY PORTED in gzippy LINE-FOR-LINE
(block_fetcher.rs:737/758/66/536, chunk_fetcher.rs:528-529/1542/1561). The generic machinery is
NOT the defect; the SPECIFIC overshoot-reuse path is. Subagent's source-read leaned engine-induced
(the causal perturbation refined this to MIXED — perturbation overrules attribution per CLAUDE.md).

### RE-SCOPED DESIGN (advisor-corrected): block-finder OFFSET SUPPLY, NOT insert-relocation
- REFUTED (advisor, leader-verified first-hand): "move the block-finder insert POOL-side" was a
  vendor MISREAD. appendSubchunksToIndexes (the insert) runs IN-ORDER on the orchestrator
  (GzipChunkFetcher.hpp:357); queueChunkForPostProcessing (:554-582) pool-submits only applyWindow.
  gzippy ALREADY inserts in-order (consumer_append_subchunks_vendor chunk_fetcher.rs:1750/2790,
  citing :343-357). Pool-side move would DIVERGE from vendor (guardrail violation).
- CANDIDATE LEVER: the block-finder offset SUPPLY. Vendor GzipBlockFinder::get (:117-158) returns
  the CONFIRMED offset for a known index, partition GUESS only for unknown. gzippy's strategy offers
  ONLY the guess + never RE-OFFERS once prefetched (needs_confirmed_offset never fires) => the stale
  overshoot guess-prefetch is never superseded. Faithful port = make gzippy re-target the overshot
  index at the confirmed offset once the in-order insert records it.
- OPEN MECHANISM QUESTION (advisor-mandated, answer BEFORE any build): vendor inserts in-order with
  the SAME ~1-chunk lead the 3 prior gzippy attempts had — so HOW does vendor avoid the overshoot
  cold-get? Re-derive first-hand (block-finder re-target within existing look-ahead vs interior
  reuse vs less-overshoot). The prior mechanistic story was tied to the refuted pool-side premise.

### CHECKPOINT REACHED — STOP for supervisor ratification. NO placement port, NO engine build started.
Subagents (vendor map + disproof advisor) read-only, synchronous, collected in-turn, killed (no
orphans). Guest run held by a Bash task holding the ssh; host RESTORE VERIFIED. Supervisor to
ratify: (1) the MIXED verdict (existence structural, magnitude engine-coupled, separability
unproven); (2) that the placement lever survives but its standalone payoff is smaller/engine-
entangled; (3) authorize TIER-3 placement ONLY to re-derive the vendor lead mechanism (block-finder
offset supply) + gate on STALL probe(3->1)+no-flood A/B — NOT the insert-relocation, NOT the
cache-read getIndexedChunk. Engine front unchanged (co-primary, +13.7%, gated by §2.3 bench).

## PREFETCH-HORIZON vs SATURATION DIAGNOSIS [2026-06-07, prefetch-horizon leader, HEAD 85c67474]
CHARTER plans/prefetch-horizon-diagnosis.md. Answer the anti-escape-hatch question: are the
all-`decode_NOT_STARTED` head-of-line stalls WORKER SATURATION (engine) or PREFETCH-HORIZON
too shallow (structural)? Falsifier PRE-REGISTERED before any run: plans/prefetch-horizon-falsifier.md.
Guest verified idle before; host RESTORE VERIFIED after every run; no orphan procs (subagents killed).

### INSTRUMENT (commit 85c67474, byte-exact, env-gated) — STALL_OCCUPANCY_PROBE
At each non-startup cold-get stall (chunk_fetcher.rs:1357 None branch), snapshot worker occupancy
(thread_pool busy/idle_capacity via new idle_thread_count() accessor) + whether the stalled index
was enqueued (in-flight key covers decode_start). Classifies SAT (idle_cap==0) / HORIZON_NOT_ENQUEUED
(idle_cap>0 & not enqueued) / HORIZON_ENQUEUED_NOT_DONE. Gated on GZIPPY_STALL_RESIDENCY_PROBE
(OFF==identity); stall_residency mod now #[cfg(parallel_sm)] (was dead under default clippy).

### DATA (locked guest, silesia-large 503MB, T8, N=7-9, sha-verified, RUN_TRUSTWORTHY=true, diverged=0)
| combo | wall | SAT | HZ_NOT_ENQUEUED | mean_busy/8 | mean_idle_cap |
|---|---|---|---|---|---|
| F0 baseline | 1.125s | 1 | 2 | 5.3-6.0 | 2.0-2.67 |
| F100 spin | 1.413s | 1 | 2 | 5.67 | 2.33 |
| F100 sleep | 1.254s | 0 | 3 | 6.33 | 1.67 |
Residency (same stalls): NOT_RESIDENT=4 has_nearest_le_start=0 (never-retained / consumer-pace).
Source-cite (read-only subagent): horizon DEPTH is VENDOR-IDENTICAL (2·P candidate, P-1 concurrent,
same ramp + 1ms pump). block_fetcher.rs:737/763 ↔ BlockFetcher.hpp:467/474.

### VERDICT: NEITHER clean saturation NOR a confirmed horizon-DEPTH fix. (My first-draft "SATURATION→engine" was REFUTED by the disproof advisor; sustained.)
- SATURATION is DISPROVED: idle_capacity>0 at EVERY stall (1.67-2.67, never 0 even under 2x slow);
  a free worker existed; the on-demand decode submits onto it immediately (chunk_fetcher.rs:1450-1468).
- My slow_knob "decisive" cross-check was CONFOUNDED: engine-slow raises busy GLOBALLY regardless of
  stall cause (both hypotheses predict busy↑); the DISCRETE SAT bucket actually went 1→1→0 (falling)
  while HZ_NOT_ENQUEUED rose 2→3 — by my own pre-registered rule that is the HORIZON signature.
- I OVERRODE my own pre-registered map (HORIZON 3/3 rows) with a post-hoc continuous metric — the
  exact "attribution forecloses measurement" the falsifier forbids. Process violation, advisor-caught.
- BUT it is also NOT a confirmed horizon-DEPTH fix: a single snapshot can't tell NEVER-DISPATCHED
  (with idle cap = real scheduling/horizon gap) from DISPATCHED-DECODED-EVICTED-before-arrival
  (retention/anti-overrun, engine-lag-coupled). Both give NOT_RESIDENT+!enqueued. UNRUN discriminator.
- Engine is a genuine CO-lever but via engine-lag→cache-overrun→eviction (a corrected mechanism that
  REOPENS the placement/retention sub-question), NOT via "no free worker." NO engine redirect (that
  was the escape hatch). NO fix attempted (sub-cause unresolved). NO inline-ASM build started.

### OWED before deciding (advisor §6, the cheap unrun discriminator):
(i) per-stall NEVER-DISPATCHED vs DISPATCHED-THEN-EVICTED (track if a covering task was submitted
earlier + when evicted vs this arrival); (ii) split idle_capacity PARKED vs UNSPAWNED; (iii) N≫3
(lower split_chunk_size / aggregate dozens of stalls). Then saturation-vs-horizon is DECIDED, not
attributed. Distinct from the refuted offset-supply lever and from raw inner-loop speed.

### CHECKPOINT — STOP for supervisor gate. Full findings: plans/prefetch-horizon-findings.md;
advisor verdict: plans/prefetch-horizon-advisor-verdict.md. Subagents (source-cite + disproof
advisor) read-only/synchronous/killed (no orphans). Guest run from Bash tasks holding the ssh;
host RESTORE VERIFIED. Commit 85c67474 (probe) pushed to reimplement-isa-l. NO placement port,
NO engine work, NO inline-ASM build started.

---
## SUPERVISOR ROUTING 2026-06-07 (strategic advisor aa214edf): PIVOT TO ENGINE BENCH
After 3 placement diagnostic rounds (offset-supply refuted; saturation disproved; horizon-depth
vendor-identical), strategic advisor verdict = PIVOT-TO-ENGINE-BENCH. The engine is the
unbounded TIE-DETERMINING unknown (2.38× clean gap); placement is bounded +13.7% w/ diminishing
returns. NEXT = §2.3 engine isolation bench (the only legit ceiling-bound, isolation oracle +
ISA-L positive control), bundling 3 near-free riders: (a) re-read nearest_le_start @cap=256
(closes placement never-retained-vs-evicted); (b) same-sink production-output floor (bench can't
reach it; 0.61s was /dev/null); (c) bench self-test first. DROP "placement dissolves w/ faster
engine" as load-bearing (only ~24% coupled, cold-gets structural). PRE-REGISTER the plateau
falsifier before any engine build. See plans/engine-bench-authorization.md.

## ENGINE ISOLATION BENCH — BUILT + RUN (2026-06-07, engine-bench leader, HEAD 249f25b5)
Falsifier pre-registered: plans/engine-bench-falsifier.md. Bench: benches/engine_isolation.rs
(uncommitted; + Cargo.toml [[bench]] stanza + src/lib.rs one doc-hidden re-export
`isal_decompress_oracle` — measurement-only, decode graph untouched). Build combo
`--features pure-rust-inflate,isal-compression` on x86_64 = BOTH variant-(i) marker_inflate
clean loop (pure_inflate_decode) AND variant-(iii) ISA-L FFI in one binary; isal_clean_tail
stays OFF (gzippy-isal unset) so production routing is unaffected (build.rs:94-101, verified).

### GUEST RESULT (authoritative — native x86_64, freq-locked no_turbo=1/perf, single core 0,
### interleaved best-of-11, host RESTORE VERIFIED). Driver scripts/bench/run_engine_isolation.sh.
```
ENGINE_BENCH start_bit=302012944 N_bytes=4194304 (one clean mid-stream silesia chunk, 32KiB window)
VAR_I   scalar_u16  med=0.035545s  MBps_med=118  sigma=0.4%   (production marker_inflate::Block clean loop)
VAR_II  E1_u8(part) med=0.033563s  MBps_med=125  sigma=0.2%   (u8-direct sink; ring still u16 — bounds OUTPUT-traffic only)
VAR_III isal        med=0.010814s  MBps_med=388  sigma=6.9%   (ISA-L isal_inflate oracle, FFI measurement-only)
RATIO ii/i=1.059  iii/i=3.287  ii/iii=0.322
SHA_ALL_EQUAL=yes  CRC i=ii=iii=0xf24393b3  (BYTE-EXACT GATE PASSES every iter)
```
- BYTE-EXACT is solid: all three CRC identical over 4 MiB, sub-1% sigma on both Rust paths.
  `block.contains_marker_bytes()==false` asserted ⇒ variant (i) is a GENUINE clean decode of
  the production loop, no shortcut, not the markered arm. Bench genuinely isolates single-thread
  inner clean-decode compute (no scheduler/publish/marker machinery).
- (iii)/(i) = **3.287×** on the guest (LARGER than the prior campaign's 2.38× wall / ~2.1× s-t
  framing). This is the PURE single-thread clean inner-loop gap (one chunk, no pipeline). The
  prior 2.38× was clean-per-chunk-rate at the wall (92.7ms vs 39ms) which carries pipeline/drain;
  the isolated inner loop gap is wider.
- SELFTEST band [1.7,2.6] reads FAIL because 3.287 > 2.6 — but this is NOT a broken-instrument
  signal (byte-exact + clean + sub-1% sigma); it is ISA-L beating our scalar clean loop by MORE
  than the band. Band was a prior-number sanity range; the qualitative "ISA-L ~2-3× our scalar"
  reproduces and EXCEEDS it. (Advisor to adjudicate whether 3.29× indicates a real wider gap or a
  bench artifact — see checkpoint.)
- E1 (u8-direct OUTPUT) buys only +6% (ii/i=1.059). FINDING: output write-width is a SMALL
  component of the engine gap. The dominant ISA-L advantage is in the INNER HUFFMAN DECODE LOOP
  (multi-symbol packing / asm), not output traffic. E1-partial keeps the u16 ring; a full E1
  (u8 ring) would also cut ring traffic, but this bounds the output-side lever as minor.

### RIDER 1 — ANSWERED (from committed cap-swept evidence, commit d764734c): has_nearest_le_start=0
at ALL caps including cap=256 ⇒ NEVER-RETAINED / consumer-pace, NOT capacity-eviction. Interior-
reuse (getIndexedChunk) port has no resident parent at any cache size. Placement never-retained-vs-
evicted sub-question CLOSED.

### RIDER 2 — DONE (same-sink, real file sink on /dev/shm, T8 silesia-large, N=9, freq-locked,
### all sha OK, host RESTORE VERIFIED). Driver scripts/bench/run_same_sink_floor.sh.
```
gzippy_same_sink:    min=1.1915 med=1.2113 sd=1.3%
rapidgzip_same_sink: min=0.5960 med=0.6042 sd=1.8%
```
- **rapidgzip's SAME-SINK wall = 0.604s, NOT 0.54s.** The §3 "0.54s" was the /dev/null number;
  on a REAL file sink rapidgzip is 0.604s. The tie BAR moves UP to ~0.60s — slightly MORE
  favorable to gzippy's reachability than the design feared. (/dev/shm is RAM-backed; a true
  spinning-disk sink would lift both further — this is a fair apples-to-apples same-sink compare.)
- gzippy same-sink 1.21s ≈ 2.0× rapidgzip — consistent with the ~2.1× baseline.
- §3 contingency "same-sink floor ≤0.54s" RESOLVED: the comparison floor is rapidgzip's own
  0.60s same-sink; gzippy's non-decode serial floor (~0.015s, step-0) + same-sink output is well
  under that, so the consumer does NOT structurally forbid the tie at the corrected ~0.60s bar.

### PLATEAU-FALSIFIER VERDICT (pre-registered plans/engine-bench-falsifier.md):
- (ii) E1-partial = ii/iii = 0.322 (still ~3× slower than ISA-L). Per the pre-registered
  thresholds: (ii)/(iii)=0.322 ≤ 0.65 ⇒ E1-OUTPUT plateaus near the scalar pure ceiling, FAR
  below ISA-L, residual ≫ spread (sub-1% sigma). **E1 (output write-width) alone is NOT PROVEN**
  as the engine lever. BUT this is only the OUTPUT-traffic sub-lever; E2 (SIMD back-ref copy) /
  E3 (packed multi-literal store) / E4 (wide refill) — the INNER-LOOP techniques where the finding
  says the gap actually lives — were NOT benched (out of scope for the first subagent). So the
  PLATEAU verdict is: **E1 plateaus (output traffic is a minor +6% lever); the engine front is
  NOT YET PROVEN because the inner-loop techniques (E2-E4, the dominant gap) are unmeasured.**
  The bench is BUILT and the harness is proven (byte-exact, freq-locked, self-consistent) — it is
  ready to measure E2-E4 if the supervisor authorizes that next bench round.
### ADVISOR VERDICT (independent, read-only, disproof-driven — plans/engine-bench-advisor-verdict.md):
- (a) BENCH VALIDITY = **CORROBORATE-WITH-CAVEATS.** Variant (i) traced to the PRODUCTION LUT
  multi-cached/TRIPLE_SYM clean loop (marker_inflate.rs:968→981→1114→1191→1206), NOT a slow
  fallback. ISA-L oracle does the same work (raw deflate, same dict, same N via cap, no
  short-circuit) and is biased AGAINST itself (zeroes a 4MiB buffer INSIDE the timed region,
  isal_decompress.rs:372 ⇒ the gap is CONSERVATIVE). Byte gate is full memcmp (bench:245-247),
  not just CRC. The bench is a TRUSTWORTHY instrument for what it measures.
- (b) THE 3.29× HEADLINE = **REFUTED → corrected to ~3.10×.** Variant (i) carries a u16-sink
  re-widen tax (Vec<u16> uses the DEFAULT push_clean_u8) that PRODUCTION DOES NOT PAY —
  production uses the u8-direct clean-tail sink (gzip_chunk.rs:842), which variant (ii) mirrors.
  That tax is only ~6%. So the production-representative ceiling = **(iii)/(ii) ≈ 3.10×**, not
  (iii)/(i)=3.29×. ~3.1× is PLAUSIBLE for ISA-L-AVX2-vs-scalar-u16 (libdeflate/ISA-L beat naive
  scalar 2-4×), NOT a broken-instrument flag. It is LARGER than the prior 2.38× because the
  DENOMINATORS DIFFER: 2.38× was gzippy-clean vs rapidgzip-SYSTEM per-chunk; 3.1× is
  gzippy-current-clean-loop vs PURE-ISA-L-clean (purer denominator ⇒ larger ratio). Do NOT
  conflate them. The SELFTEST "FAIL" is a MIS-CALIBRATED band (set from the 2.38× system number),
  not a broken instrument — recalibrate to ~[2.5×,3.6×] and re-pass (rule 4).
- (c) PLATEAU / NOT-PROVEN = **CORROBORATE (sharpened).** Cannot reject (rule 3/7: E1 was never
  the closing lever; the u16 ring is still u16 so E2-E4 untested) and cannot declare proven. The
  bench built only E1-partial, but the pre-registered falsifier is defined over (ii) AFTER E2-E4.
- (d) RIDER-2 FLOOR = **CORROBORATE-WITH-CAVEATS.** /dev/shm is a legit same-sink ratio proxy
  (neither tool fsyncs); bar moves 0.54→0.604s (favorable) but gzippy at 2.0× doesn't change the
  verdict; the ~2× same-sink wall is consistent with the ~3.1× isolated inner-loop gap.

### THREE OWED ITEMS before this number can authorize the multi-week engine build (advisor):
1. Re-quote the ceiling as **(iii)/(ii) ≈ 3.10×** (production-representative), not 3.29×. [DONE here.]
2. **Sweep ≥3-5 chunks** (text + binary regions) and report the range — one chunk is suggestive,
   not a ceiling bound.
3. **Build the E2-E4 prototype in variant (ii)** and re-run — the pre-registered falsifier
   adjudicates (ii)-AFTER-E2-E4, which were NOT built this round.

### CHECKPOINT — STOP for supervisor gate. NET: the engine isolation bench is BUILT, freq-locked-
### guest-validated, byte-exact, and advisor-confirmed TRUSTWORTHY. The PRODUCTION-REPRESENTATIVE
### single-thread clean inner-loop ceiling = **gzippy ~3.1× slower than pure ISA-L** (118-125 MB/s
### vs 388 MB/s). E1 (output write-width) plateaus at +6% — output traffic is a MINOR lever; the
### gap lives in the INNER HUFFMAN LOOP (E2 SIMD copy / E3 packed store / E4 wide refill), which
### remain UNMEASURED. PLATEAU FALSIFIER: NOT-PROVEN-YET (E2-E4 unbenched), NOT refuted. Same-sink
### tie bar corrected up to ~0.60s. Riders both answered. Per the charter: DO NOT start the
### multi-week production engine integration/build. The decision the supervisor gates: authorize a
### SECOND bench round (sweep chunks + prototype E2-E4 in-bench) to settle the plateau falsifier,
### OR re-confront placement. No production code was changed (bench files uncommitted; lib.rs has a
### measurement-only re-export). HEAD unchanged at 249f25b5.

## INNER HUFFMAN KERNEL — speculative-pipeline isolation bench [2026-06-07, leader fresh instance]
Charter: plans/inner-huffman-kernel.md. State (causal, advisor-upheld): faithful u8 LANDED
byte-exact (fa9fd73c), wall TIE, gap constant ~1.70x at T1+T8 => binder is per-symbol Huffman
LUT-decode COMPUTE. Traffic/ring/placement all SLACK.

### SOURCE-VERIFIED FIRST-HAND (before any code):
- igzip trick #1 (packed-flat-short-code table, one u32 retires up to 3 literals + bit-length)
  is ALREADY BUILT + PRODUCTION-LIVE: lut_huffman.rs make_inflate_huff_code_lit_len builds with
  TRIPLE_SYM_FLAG=0 (=> triples ENABLED); LutLitLenCode::rebuild_from (lut_huffman.rs:998) passes
  it; the production clean loop read_internal_compressed_specialized<false> (marker_inflate.rs:1475)
  decodes via lut_litlen_decode -> (symbol, sym_count, bit_count) and unpacks up to 3 packed
  literals (marker_inflate.rs:1492-1524). So trick #1 is NOT the remaining lever.
- igzip trick #2 (speculative software-pipelined loop) is NOT present in production: the clean
  loop (a) stores ONE u8 per inner iter with `pos % U8_RING_SIZE` (marker_inflate.rs:1504), NOT an
  unconditional 8-byte packed store advanced by actual count; (b) does NOT preload next lit/len +
  dist before knowing current symbol type; (c) refills every outer iter. THIS is the remaining
  lever: the speculative pipeline (igzip asm:507-627) on the flat-u8 path.

### PRE-REGISTERED FALSIFIER (set BEFORE building/measuring):
- BUILD: a new flat-u8 bench variant in benches/engine_isolation.rs (VAR_V) that decodes the SAME
  clean silesia chunk via the EXISTING packed-flat-short-code table (LutLitLenCode/LutDistCode,
  trick #1) + a NEW speculative software-pipelined loop (trick #2): unconditional 8-byte packed-
  literal store into a flat linear u8 buffer advanced by actual sym_count, slop-margin headroom
  guard (no per-symbol bounds check inside the fast region), preload of next litlen+dist, branchless
  multi-literal output, MOVDQU/word back-ref copy on the flat buffer. Pure-Rust first; inline-asm
  only where Rust codegen demonstrably lags (measured, not assumed).
- BYTE-EXACT GATE (absolute): VAR_V SHA-equal vs VAR_I scalar AND vs VAR_III ISA-L over the swept
  clean chunks, all 5. A wrong-bytes oracle is VOID.
- SELF-TEST: VAR_III/VAR_I (iii/i) in [2.5,3.6] on the guest (existing band) must still hold —
  validates the ISA-L oracle denominator.
- PASS  = VAR_V clean MB/s (guest, median-of-per-chunk-medians) gives (V)/(iii ISA-L) >= 0.85,
  i.e. via tier1-design v2 §3 (anchored 92.7ms<->104MB/s, 39 chunks, ramp 1.36, T8) projects
  decode_wall that RE-BINDS off decode (<= 0.604s same-sink bar + spread), closing the ~1.70x.
- PLATEAU = (V)/(iii) stays ~pure class (<= 0.65, the round-2 plateau line) with residual to 0.85
  > inter-chunk spread (round-2 sd ~0.036). Per the standing falsifier this re-opens the FFI/bar
  fork (is igzip-class reachable in pure-Rust+asm AT ALL once BOTH tricks are present).
- Numbers ONLY from the locked guest (interleaved, taskset-pinned, sha-verified). Rosetta gives
  byte-exact validation but NOT authoritative MB/s.

### RESULT — AUTHORITATIVE GUEST (freq-locked, AVX2 live, SHA_ALL_EQUAL=yes, N=11 interleaved, taskset core0)
Host driver run_engine_isolation.sh -> host_lock_and_bench.sh (no_turbo pin + RESTORE VERIFIED).
ALL 8 variants BYTE-EXACT on all 5 swept clean silesia chunks (SHA_ALL_EQUAL=yes incl VAR_V; the
VAR_IV/VAR_V Rosetta VOIDs during local iteration were Rosetta-x86-64-v2 artifacts, NOT real —
the guest native run is byte-clean). Self-test PASS (iii/i=2.76, in [2.5,3.6]).

AGGREGATE (median-of-per-chunk-medians, MB/s):
  VAR_I scalar (production clean loop)              105   vs_isal 0.374
  VAR_IV_E234 (round-2 grafts, u16 ring)           116   vs_isal 0.413
  VAR_V packed-table + SPECULATIVE PIPELINE flat-u8 156   vs_isal 0.555  <-- THE LEVER
  VAR_III ISA-L oracle                             281   vs_isal 1.000
Per-chunk VAR_V vs_isal: 0.554 0.551 0.555 0.555 0.511 (median 0.554, sd ~0.018, tight).

VERDICT = PLATEAU (by the PRE-REGISTERED structural falsifier).
- PASS line (pre-registered) = (V)/(iii) >= 0.85 so decode RE-BINDS off the shared pipeline floor.
  Measured 0.554. Residual to 0.85 = 0.296 ~= 16 sd. FALSIFIER FIRES -> PLATEAU.
- BUT a MATERIAL ADVANCE over the round-2 plateau: 0.413 -> 0.554 (+34% over E234, +48% over
  scalar). The speculative software-pipelined loop (trick #2) on flat-u8 is the single biggest
  inner-loop gain measured this campaign. It confirms (a) trick #1 packed-table was ALREADY
  production-live (so the round-2 "grafts" weren't the table), and (b) the genuine remaining
  lever was trick #2 + flat-u8 linear buffer, which VAR_V isolates and DELIVERS — but to 0.55x,
  not igzip-class.
- §3 projection (anchored 92.7ms<->104MB/s, 39 chunks, ramp 1.36, T8): VAR_V projects decode_wall
  ~0.410s. This is NUMERICALLY < the 0.604s tie bar, AND now also < the ~0.54s placement-laden
  consumer floor -> at 0.55x ISA-L the decode term FALLS BELOW the floor, so decode would stop
  being the SOLE binder and the wall would RE-BIND on the ~0.54s floor. This is exactly the round-2
  "numeric pass is an artifact while the RATIO criterion governs" situation — the pre-registered
  RATIO criterion (0.85) GOVERNS -> PLATEAU. (The decode-falls-below-floor observation is a real
  shift worth the advisor + supervisor's attention: it suggests at VAR_V's rate the BINDER may move
  from engine to the placement/consumer floor — the co-primary from project_pregate.)
- Per the standing falsifier, PLATEAU re-opens the FFI/bar fork: igzip-class (0.85) is NOT reached
  in pure-Rust even with BOTH tricks (packed table + speculative pipeline) on flat-u8. Remaining
  gap to ISA-L (0.55->1.0) is plausibly the inline-ASM/BMI2-specific codegen + the unchecked
  over-read/write the safe-Rust headroom guard only partially exploits. Inline-asm is authorized
  but is the next, larger increment; this checkpoint settles the PURE-RUST ceiling of the lever.
- DECISION: do NOT integrate yet (pre-registered: integrate only on PASS). Report the achievable
  pure-Rust ceiling (0.55x) + the decode-falls-below-floor nuance to the supervisor. inline-asm
  spike + the placement co-primary are the two named next directions.

## [2026-06-09] STRUCTURAL-RESIDUAL SIZING CHECKPOINT (owner/structural-residual-sizing @ d56cb0f5, bin 2d317027)
Full detail + raw numbers appended to plans/disproof-ledger.md (same date). Summary:
- STEP-0: gzippy-isal at HEAD, isal_chunks=14@T4 / 16@T1 fb=0/1, path=ParallelSM, bin_sha 2d317027
  (== BAR-2 banked binary). Frozen guest, interleaved N=13-15, sha-verified.
- (b) SERIAL-OUTPUT = PROVED SHARED FLOOR (dual-removal oracle: gz 51/86ms ≈ rg 44/85ms @T4/T1;
  removing output from both leaves ratio unchanged 0.86-0.88). NOT a lever (~0% gz-excess).
- (a) MARKER-BOOTSTRAP = ON the T4 critical path (slow-inject spin 68/155ms @50/100%, sleep control
  23/67ms, T1 self-test FLAT 16/8ms) but SHARED with rg (gz marker compute ~59-139ms > the ~51ms gap
  => removing it overshoots rg; T4 output-removed ratio ≈ T1's). NOT the net gz-excess lever.
- (c) CHUNK-0/PER-CHUNK/PIPELINE = the gzippy-SPECIFIC excess: T1 engine-matched(ISA-L) output-removed
  gz 932 vs rg 808 = 124ms/0.867x (~100% of the output-removed gap). CANDIDATE FAITHFUL-PORT LEVER
  (rg per-chunk window-map + consumer pipeline), NOT removal-proved irreducible. Owes its own oracle.
- VERDICT: BAR-1 low-T is NOT a single proved floor; output+marker are shared/non-levers; the per-chunk
  pipeline is the live lever. "native >=0.99 every T unreachable" stays LEADING HYPOTHESIS, not proven.
- Tooling: scripts/bench/{residual.sh,_residual_guest.sh} (new; flag to Steward). Box released clean,
  no orphans (local+guest+neurotic swept). OWES supervisor Opus gate + Steward bankability.

## D1 OUTPUT-OVER-RESERVE CONVERGENCE — MEASURED, REFUTED as the per-byte lever [2026-06-09, owner/isal-d1-reserve, branch perf/isal-d1-reserve @ dc14ba36, bin cebb7a43]

CHARTER: plans/isal-perbyte-convergence.md (JOB-1 D1 over-reserve + JOB-2 glue). The
capstone (structural-residual-capstone-verdict.md) named D1 "8× output over-reserve"
as one of two suspects for the ~13% per-byte T1 gap (gz 226 vs rg 261 MB/s, identical
ISA-L kernel).

STEP-0 PREMISE (rg uses ISA-L at T1) — VERIFIED FIRST-HAND: rapidgzip 0.16.0's
compiled `rapidgzip.cpython-313-x86_64-linux-gnu.so` STATICALLY links ISA-L —
`nm` shows `decode_huffman_code_block_stateless_01`/`_04` (the AVX2/AVX512 igzip
inflate kernels) + the string "Time spent decoding with ISA-L". So rg's T1 inflate
IS the same igzip kernel gzippy's FFI calls — engine-matched premise HOLDS.

STEP-0 PROOF-OF-BINARY: freshly built gzippy-isal at HEAD (feature gzippy-isal,
x86_64, RUSTFLAGS target-cpu=native, cargo-lock serialized), env-unset:
path=ParallelSM, isal_chunks=14 fb=0 @T4 / 16 fb=1 @T1, sha=028bd002… == pin.
Real-ISA-L fingerprint (native stub can never increment). bin_sha cebb7a43.

CHANGE: env-gated `GZIPPY_ISAL_RESERVE_FACTOR` (gzip_chunk.rs `isal_reserve_factor`),
DEFAULT 8 == byte-/cost-identity to HEAD; harness scripts/bench/{d1.sh,_d1_guest.sh}
(interleaved, freeze-locked, per-factor isal_chunks/fb readback, sha-verified every
arm, + perf-stat page-fault pass). Reuses the parity/residual contamination bar.

NUMBERS (frozen guest "trainer" REDACTED_IP, bench-lock no_turbo=1 gov=performance,
quiet-gate runnable_avg<=2.0, interleaved N=11, /dev/shm regular-file sink, sha=OK
every arm, vs rg 0.16.0):

  T4 (mask 0,2,4,6), rg 498ms:
    factor= 8 (HEAD): wall 551ms  ratio_vs_rg 0.903  faults 105,246  fb 0
    factor=12       : wall 570ms  ratio_vs_rg 0.874  faults 133,822  fb 0  (+18ms,+27% faults)
    factor=16       : wall 558ms  ratio_vs_rg 0.891  faults 133,849  fb 0  (+7ms)
  T1 (mask 0), rg 925ms, gz spreads 0.8-1.8% (TIGHT):
    factor= 8 (HEAD): wall 1025ms ratio_vs_rg 0.903  faults 56,946   fb 1*
    factor=12       : wall 1024ms ratio_vs_rg 0.904  faults 56,947   fb 1*  (-1ms, faults IDENTICAL)
    factor=16       : wall 1024ms ratio_vs_rg 0.903  faults 56,944   fb 1*  (-1ms, faults IDENTICAL)
  (* T1 fb=1 is pre-existing HEAD behavior — one window-absent bootstrap chunk
     legitimately declines; SAME for all factors => my change is identity at default.)

FALLBACK FLOOR (the load-bearing structural fact): factors 5/6/7 EACH force exactly
1 ISA-L->pure-Rust fallback (isal_chunks 14->13) — one silesia chunk genuinely
decodes at ~7.5× its compressed span. So **factor 8 is the TIGHTEST 0-fallback
UNIFORM factor**: the reserve is NOT freely reducible without per-chunk fallbacks
(which void engine-matching). The "8× over-reserve" is real on AVERAGE chunks
(~3.3× ratio => 2.4× over) but is pinned by the worst chunk.

VERDICT — D1 is NOT the per-byte lever (REJECTED with mechanism, rule 7b):
1. T1 (the clean per-byte cell: engine-matched ISA-L, no marker bootstrap, single
   reused buffer): reserve size is wall- AND fault-NEUTRAL — page-faults IDENTICAL
   to the byte (56,946/56,947/56,944), wall Δ≤1ms across 8/12/16. Lazy faulting
   touches only WRITTEN pages; the written footprint == decoded bytes regardless of
   capacity. The over-reserve costs nothing here. The capstone "over-reserve =>
   cache/TLB pressure => slows the kernel even at T1" hypothesis is DIRECTLY REFUTED.
2. T4: bigger reserve ADDS faults (105k->134k, rpmalloc multi-buffer span-cache
   spill when 48-64MB spans exceed the per-thread cache) but buys only +7..18ms wall
   => faults are largely SLACK (~sub-ms wall per 1000 faults). Going below 8 is
   impossible (fallbacks). So even the FAITHFUL incremental-growth fix (per-chunk
   adaptive: avg chunk reserves ~its 13MB decoded not 32MB) has a CEILING of a few
   ms (T1 fault-neutral + T4 low fault->wall slope) — far below the ~50ms T4 / ~97ms
   T1 gaps. Not worth the invasive regrow-resume change.
=> Converging D1 does NOT move isal T1/T4 toward 0.99.

JOB-2 GLUE (source+disasm-verified; tractable attempts assessed):
- INNER KERNEL IDENTICAL: gzippy vendor/isa-l vs rg vendor/.../external/isa-l
  igzip_inflate.c differ only cosmetically (const qualifiers, NO_CHECKSUM ifdefs);
  the stopping-point patch is byte-for-byte identical. The inflate HOT LOOP is
  hand-written NASM (igzip_decode_block_stateless_01/04.asm) — same in both.
- avail_out FEED: REFUTED as a divergence. rg's IsalInflateWrapper::readInto
  (isal.hpp:258) sets avail_out to the WHOLE output buffer and loops, same as
  gzippy hands the whole reserve. "rg refills in small BitReader chunks" is FALSE
  (that's input-side). avail_out-amortization is NOT a real lever.
- CROSS-LANGUAGE LTO: WRONG TOOL + high-effort/low-payoff => NOT attempted (per
  charter "if LTO is a yak-shave, STOP+report"). The hot kernel is NASM; LTO/IPO
  cannot optimize asm. LTO could at most inline the C `isal_inflate` dispatcher
  into the caller — a per-CALL saving (~thousands of ns total), NOT per-byte. AND
  gzippy's ISA-L is built by isal-sys via autotools ./configure+make (a separate
  .a, NASM .o + gcc C); emitting LLVM bitcode for cross-lang LTO would be a
  multi-hour autotools->cc-rs+bitcode rewrite for a per-call-only payoff. Feasibility:
  HIGH effort, ~ZERO per-byte payoff. STOP.
- RESIDUAL per-byte (~0.097x @T1): the inner asm kernel is identical and the
  per-call/per-block FFI deltas are sub-ms, so the gap is NOT glue/reserve/LTO. It
  lives in the broader PER-CHUNK + ParallelSM PIPELINE that gzippy runs even at T1
  (16 separate ISA-L invocations w/ init+set_dict(32KB window copy)+boundary loop,
  ring/window-map/CRC/handoff) vs rg's leaner per-chunk consumer. This == the
  capstone's "per-chunk/pipeline" term = an ARCHITECTURE-port lever (faithful port
  of rg's consumer/window-map), NOT a glue/reserve/LTO quick fix.

ARTIFACTS: /tmp/d1_T1.log, /tmp/d1_T4.log, /tmp/d1_build_parity.log (host side).
Branch perf/isal-d1-reserve @ dc14ba36 (worktree). Box released clean, no orphans
(local/guest/neurotic pgrep clean). OWES: supervisor Opus gate on the D1-rejection
inference + the per-chunk/pipeline-is-the-residual attribution (NOT removal-proved
here — only D1 + glue eliminated; the pipeline term still owes its own oracle).

## [2026-06-09] PER-CHUNK / PARALLELSM-PIPELINE ISOLATION — REMOVAL-ORACLE VERDICT (owner/perchunk-singleshot, branch perf/perchunk-singleshot, bin 9c466f67, worktree .claude/worktrees/perchunk-singleshot)
CHARTER: plans/perchunk-pipeline-isolation.md. The last isal low-T suspect after DIS-14
removal-eliminated D1+glue and RE-LOCALIZED to the per-chunk ParallelSM pipeline (which
still owed its own oracle). This run SUPPLIES that oracle and CLOSES the isal low-T attribution.

ORACLE (existing single-shot path — source-verified first): `isal_decompress::decompress_gzip_stream`
(isal_decompress.rs:25) already decodes a WHOLE gzip stream in ONE ISA-L call (no chunking, no
per-chunk set_dict(32KB), no ring/window-map/handoff/per-chunk-CRC). Wired a MEASUREMENT-ONLY env
gate `GZIPPY_ISAL_SINGLESHOT=1` (`try_isal_singleshot_oracle`, src/decompress/mod.rs) at all three
CLI single-member entries (decompress_gzip_libdeflate, decompress_single_member, _fd). NOT
production (env-gated, isal-only, byte-exact). Harness scripts/bench/{perchunk.sh,_perchunk_guest.sh}
(reuses parity/residual contamination bar: GZIPPY_* scrub, stale-binary fingerprint, ParallelSM
assert + isal_chunks readback for gz-prod, IsalSingleShot path assert for the oracle, same-sink
regular file, interleaved, sha-verify every arm).

STEP-0 proof-of-binary: gzippy-isal at HEAD-equiv (worktree), RUSTFLAGS target-cpu=native,
cargo-lock serialized. gz-prod env-unset-but-FORCE_PARALLEL_SM=1: path=ParallelSM,
isal_chunks=16 fb=1 @T1 / 14 fb=0 @T4. gz-singleshot: path=IsalSingleShot(oracle). sha=028bd002… OK.

NUMBERS (frozen guest REDACTED_IP, bench-lock no_turbo=1 gov=performance, quiet runnable_avg=1.00,
interleaved N=15, /dev/shm regular-file same-sink, sha=OK every arm, vs rg 0.16.0):
  T1 (mask 0, spreads 0.6-1.0% — TIGHT):
    gz-prod (ParallelSM 16-chunk) : 1.0131s  209 MB/s  ratio_vs_rg 0.905
    gz-singleshot (1 ISA-L call)  : 0.7659s  277 MB/s  ratio_vs_rg 1.197   <== BEATS rg
    rg-file                       : 0.9164s  231 MB/s
    => per-chunk pipeline cost = gz-prod - gz-singleshot = 247 ms (~24% of T1 wall)
    => component: perchunk init+set_dict = 124 us over 17 calls = 0.05% of the 247 ms
  T4 (mask 0,2,4,6, spreads 23-29% — jittery):
    gz-prod (ParallelSM 14-chunk) : 0.5387s  393 MB/s  ratio_vs_rg 0.911
    gz-singleshot (single-thread) : 0.7543s  281 MB/s  ratio_vs_rg 0.651  (can't use 4 cores)
    rg-file                       : 0.4910s  432 MB/s
    => pipeline is net-POSITIVE +216 ms at T4 (it BUYS the parallelism)

DECOMPOSITION (GZIPPY_VERBOSE counters @T1, gz-prod): markers are ZERO (flip_to_clean=0
finished_no_flip=0), window_seeded=16, finish_decode=17, inflate_wrapper=1, isal_fallbacks=1.
So the 247 ms is NOT init/set_dict (124 us) and NOT marker resolution (0 markers) — it is the
chunk-LIFECYCLE: 1 pure-Rust fallback re-decode + ring/window-map/CRC-per-chunk/handoff + the T1
SERIALIZATION (each chunk waits the prior chunk's 32 KB tail-window before ISA-L can decode =>
fully serial with handoff latency, and ZERO parallelism benefit at T1).

VERDICT (pre-registered falsifier RESOLVED): single-shot ISA-L @T1 = 1.197x rg => the per-chunk
ParallelSM pipeline IS the entire isal low-T gap (an ARCHITECTURE/ROUTING lever; rg = existence
proof, and the lever's CEILING overshoots rg by 20%). The competing "isal low-T is a proved FLOOR /
the in-process ISA-L call" hypothesis is REFUTED (DIS-15) — single-shot uses the SAME igzip kernel
and is the FASTEST arm. This COMPLETES the isal low-T attribution: output (DIS shared floor) +
marker bootstrap (shared) + D1/glue (DIS-14) + per-chunk pipeline (DIS-15, THIS) — only the last is
the gzippy-specific low-T excess, and it is a real, sized lever. The cheapest realization is a
ROUTING fix: at T1 route to single-shot ISA-L instead of forcing chunking. The T4 0.911x residual is
a SEPARATE parallel-scheduling gap (single-shot can't help there).

CAVEAT owed to the gate: gz-prod is FORCE_PARALLEL_SM=1 (the campaign's standing T1 convention,
apples-to-apples with prior BAR-1 0.903x); whether real-production `-p1` forces ParallelSM is a
routing question to confirm before banking the routing fix. gzippy-native UNTOUCHED (separate 0.667x
engine floor). Box released clean, no orphans (local+guest+neurotic swept; neurotic no_turbo=0,
watchdog inactive). OWES supervisor Opus gate. Tooling flag to Steward: perchunk.sh/_perchunk_guest.sh.

## T4/T8 CONTENTION + CACHE-RESIDENCY INVESTIGATION — STEP 0 disasm CLOSED, STEP 1 contention REFUTED [2026-06-09, owner/t4-contention @ d56cb0f5, bin 378788924ace0381]

Charter plans/t4-contention-cache-residency.md. Frozen guest REDACTED_IP (bench-lock, no_turbo=1
~1.394 GHz, released clean; orphan find/ from a STEP-0 search reaped on guest+neurotic+local). Agent
tool absent => self-disproof only; OWES supervisor Opus gate. Full numbers/mechanism in disproof-ledger.md
(STEP-0 DISASM section + DIS-17).

- STEP 0: proof-of-binary isal_chunks=16/14/14 fb=1/0/0 path=ParallelSM. DISASM ISA-L Level-2 CLOSED:
  gzippy's igzip_decode_block_stateless_04.o = AVX2/BMI2 nasm kernel (5 VEX + 17 BMI2, 0 SSE; _01.o=SSE);
  a contiguous _04 byte-run is verbatim in BOTH the stripped gzippy binary AND rapidgzip's .cpython .so
  => gzippy & rg execute the IDENTICAL AVX2/BMI2 igzip kernel byte-for-byte. (rapidgzip CLI = python
  wrapper; real ELF is the .so.)
- STEP 1: DIS-17 — the contention/variance hypothesis REFUTED. Wall T4 0.898x (gz/rg spread 4%/4%),
  T8 0.985x TIE (10%/9%) — gz & rg variance MATCHED; the "17-36% gz" disparity does NOT reproduce frozen
  (thaw artifact). perf: gz +40% INSTRUCTIONS (the dominant gap = WORK, re-confirms LEV-4/VAR_VIII engine
  story); LLC & L1 MPKI EQUAL-OR-BETTER (not cache-contended); false-sharing c2c HITM=6 (noise); ctx-sw
  2.3x but tiny. ONE located memory diff = TLB/page-fault FOOTPRINT (RSS +25%, page-faults 2x, dTLB MPKI
  ~2x; on the user north star) but a SMALL wall term (~9-10ms vs 56-72ms gap) + DIS-14 already sized it
  wall-slack. STEP 2 NOT entered (no contention to converge; footprint = DIS-14 re-litigation w/o new
  mechanism => owed its own gated turn). NEW tooling for Steward: scripts/analysis/disasm_proof.sh,
  scripts/bench/{perf_contention.sh,_perf_contention_guest.sh}.

## DIS-20 (2026-06-09): DE-FRAG WALL A/B — de-frag is ON the T4 critical path (NOT wall-slack), near-slack at T1
The owed causal perturbation for DIS-19's ~1.70e9 u16 OUTPUT/BACKREF FRAGMENTATION sub-lever (the
rg-marker gate's CLAIM B FIX-NEEDED). Byte-transparent SLOW-injection (GZIPPY_SLOW_DEFRAG) at the EXACT
de-frag sites (emit_backref_ring u16-ring + marker drain push_slice), frozen guest, interleaved N=15,
sha=OK every arm, path=ParallelSM, isal_chunks=14/0@T4. CLEAN (hit-counter atomic confound caught+removed):
- T4 (OFF 551ms/0.891): DEFRAG 100/200/400% spin => 794/982/1323ms (+243/+431/+772ms, 6-19x spread);
  200% SLEEP control => 728ms (+177ms) => criticality SURVIVES freq-neutral swap. ON THE CRITICAL PATH.
- T1 (OFF 1026ms/0.898): DEFRAG 200% spin => 1063ms (+37ms) => NEAR-SLACK (T1 serialization-bound, DIS-15).
VERDICT: REFUTES "de-frag is wall-slack" AT T4 (TIE-6 footprint-slack does NOT extend to data-movement).
de-frag flat-buffer PORT is WARRANTED at T4 (largest faithful sub-lever). Rule 3: slow-slope sizes
CRITICALITY not the speed-up ceiling => flat-buffer REMOVAL oracle (or direct faithful port + measure vs
rg) is the OWED next step to size the recovered fraction of the ~60ms T4 gap. Do NOT touch flip-to-clean.
See plans/disproof-ledger.md DIS-20. OWES supervisor Opus gate. NEW tooling -> Steward.

## T1 SINGLE-SHOT ROUTE — LANDED + VERIFIED (2026-06-09, owner/t1-singleshot-route)

Production routing change: gzippy-isal + single-member + num_threads<=1 -> new
DecodePath::IsalSingleShot (one ISA-L `decompress_gzip_stream` call; CRC32+ISIZE
verified, no fallback). T>1-isal and ALL native stay ParallelSM. BGZF/multi-member
classified earlier (unaffected). This turns DIS-15's measured lever into routing.

FROZEN-GUEST PARITY (interleaved, sha-verified every trial, path-asserted; full DIS-22):
  T1  IsalSingleShot  0.766s vs rg 0.919s = 1.200x  WIN   (N=15)
  T4  ParallelSM      0.549s vs rg 0.498s = 0.906x  LOSS  (N=11, pre-existing gap, UNCHANGED)
  T8  ParallelSM      0.361s vs rg 0.374s = 1.038x  TIE   (N=11, UNCHANGED)
CORRECTNESS: 887 lib tests pass (0 fail); multi-member-at-T1 -> MultiMemberSeq byte-exact
(not swallowed); dual-sha both features at T1/T4/T8 OK; native byte-transparent.
See plans/disproof-ledger.md DIS-22. OWES supervisor Opus gate. EXPECT_PATH knob added to
parity.sh/_parity_guest.sh -> flag to Steward.

## T4-vs-T8 FULL CURVE + S/W fit + engine-vs-machinery verdict (DIS-18) [2026-06-09, owner/t4-curve @ d56cb0f5, bin b9eb0a733b4ccb6d, gzippy-isal]

QUESTION: why is gz LESS competitive at T4 (0.90x) than T8 (1.01x)? Filled the missing
thread counts; ran the H1/H2/H3 discriminators in the same campaign. Frozen guest
(no_turbo=1 gov=perf), N=13 interleaved, sha=OK every cell, path=ParallelSM asserted.

CURVE (ratio = rg_wall/gz_wall, gz forced-SM vs rg 0.16.0):
  T1 0.899 | T2 0.864(trough) | T3 0.887 | T4 0.901 | T5 0.936 | T6 0.968 |
  T7 1.002(crossover) | T8 1.011 | T9 0.873(chunk-count jump + SMT-spill confound)
  Monotonic climb, single crossover at T≈7. Scaling-to-T8: gz 2.835x vs rg 2.522x.

FIT wall=S+W/T (scripts/analysis/sw_fit.py), T3..T8 r²=.996/.986:
  W_gz=1607ms > W_rg=1188ms (+35%); S_gz=161ms < S_rg=217ms (−26%); crossover T*=7.49 ∈ (4,8).
  => H1 (Amdahl crossover) CONFIRMED on all 3 clauses. (T1..T8 fit r²=.82 — T1/T2 are
  DIS-15 serial-startup, anti-Amdahl, distort the intercept; valid only in steady regime.)

DISCRIMINATOR (CPU busy-fraction, freeze-insensitive): gz T2/T4/T8 = 94/91/72%;
  rg = 91/82/59%. At the T4 trough gz workers are 91% BUSY (not idle) and gz is MORE
  utilized than rg at EVERY T. => H2 (pool contention) and H3 (prefetch starvation)
  REFUTED (starvation would show LOW utilization). rg --verbose: 17 chunks CONSTANT
  T2/T4/T8 => H4 tail-wave REFUTED. T9 gz-regression-on-more-chunks corroborates gz is
  S-floor-bound at high T (SMT-spill co-confound flagged).

VERDICT: the T4 trough is ENGINE-W (Amdahl, asm-bounded pure-Rust marker/inner-loop
  symbol rate), NOT a machinery defect. The MACHINERY (serial floor S, scheduler
  utilization) is BETTER than vendor; the ENGINE (W) is worse. LEVER = close W
  (inner-loop symbol rate); S already at-or-below rg. Matches MEMORY's standing verdict.
  T1-bypass single-shot (DIS-15 0.766s/1.197x) NOT in d56cb0f5 (lives on owner/t1-singleshot-route).
  See plans/disproof-ledger.md DIS-18. OWES supervisor Opus gate. NEW tooling (flag to
  Steward): parity.sh pin_mask T2/3/5/6/7/9 + --bypass; _cpu_discriminator.sh; sw_fit.py.

## HIGH-T CURVE + TOPOLOGY CONTROL + per-T fb/flip counters — VERDICT: LOSS at 16+ (DIS-24) [2026-06-09, owner/t4-curve @ d56cb0f5, bin b9eb0a733b4ccb6d]

The amdahl-verdict-gate.md CLAIM-3(a) owed measurement — the goal's OWN 16+-thread regime — is RESOLVED.
One frozen-snapshot multi-T topology sweep (NEW tooling scripts/bench/hicurve.sh + _hicurve_guest.sh,
faithful parity-spine extension; build ONCE, freeze ONCE, loop cells; per-cell GZIPPY_VERBOSE counters).
i7-13700T topology VERIFIED: 8 P-cores w/ SMT (logical 0-1..14-15) + 8 E-cores no-SMT (16-23) = 24 logical
=> the old "T16=0..15" mask is SMT-oversubscribed on 8 P-cores, NOT 16 physical cores (the gate's confound).

CURVE (ratio=rg/gz, >1=gz wins; gz forced-SM vs rg 0.16.0; N=9 interleaved, sha=OK, ParallelSM):
  T8-Pphys 1.001 TIE | T9-E 0.938 | T9-SMT 0.890 | T10-E 0.855 | T12-E 0.790 | T14-E 0.768(trough) |
  T16-Ephys(16 PHYSICAL) 0.861 | T16-SMT(old mask) 0.912 | T24-all 0.889 | T32-oversub 0.893
  PEAKS at T7/T8, then TURNS OVER — gz LOSES every cell T9..T32.

PER-T COUNTERS (the gate's owed fb/flip firm-up — fb is NOT 0 and flip does NOT plateau above T8):
  chunks(finish_decode): 14→19→19→19→23→23→28→28→34→34 (T-PROPORTIONAL; rg holds ~17 CONSTANT)
  isal_fallbacks:        0 → 1 (T9..T16) → 2 (T24/T32)
  flip_to_clean:         12→18→18→18→20→20→25→25→31→31 (marker fraction RISES 12→31)

VERDICT: the goal's 16+-thread regime is a LOSS for gzippy-isal. The S-floor "gz keeps winning as T→∞"
story INVERTS past T8. HIGH-T BINDER = gz's T-proportional chunk-count growth (14→34) vs rg's constant ~17:
each added chunk raises W (more speculative marker-decode), S (longer serial publish-chain), and fb risk
(~7.5× re-decode spikes). gz over-partitions for E-cores (low IPC) / SMT siblings (shared ports) that don't
deliver P-core-equivalent throughput, while rg's wall stays FLAT (saturated/floored). gz WINS/TIES rg ONLY
in the narrow T7-T8 window. The fixed-W Amdahl fit (DIS-18) is MEASURED-contaminated above T8 and must NOT
be extrapolated to high-T. T9 dip DISENTANGLED: real machinery knee (T9-E=0.938, no SMT spill) + SMT-spill
penalty (T9-SMT=0.890) — the dip survives clean physical placement => chunk-count machinery, not topology.

OWES supervisor Opus gate (no advisor in owner env => self-disproof only). Box released clean (RESTORE
VERIFIED, watchdog inactive); leaked timeout wrappers reaped; guest+neurotic lock-free. NEW tooling ->
Steward: scripts/bench/{hicurve.sh,_hicurve_guest.sh}. See plans/disproof-ledger.md DIS-24.

## DIS-26 [2026-06-09, owner/dis26-consumer-decompose] — the pre-registered DIS-6 item (i) consumer
## decompose + CAUSAL oracle at HIGH-T (T16-Ephys, container live-expanded cores:16->24 then RESTORED).
VERDICT = (b): the high-T in-order CONSUMER is DECODE-WAIT-bound, NOT window-publish/post-process
SERIAL-WORK-bound. fulcrum_total decompose of the wall-critical consumer thread (production, sha=OK,
selftest PASS): DECODE-WAIT (blocked on worker decodes) 66%/50%/54% @ T8/T16/T24; serial writev OUTPUT
31%/43%/36%; the NAMED serial-work (window-publish + post-process on the consumer) only 1.7%/3.6%/6.9%
— and post_process.apply_window (heavy) ALREADY runs on the POOL, not the consumer; window_publish_* is
sub-ms. CAUSAL oracle (per-chunk delay knob at the publish site, gated off==identity, patched-rebuild
restored after): sleep 0/500/1000/2000us => 328/343/361/398ms (~1:1 with chunks*delay, ~20% compounding;
spin==sleep => freq-neutral). Per rule 3 the ~1:1 slope is the in-order-consumer-IS-the-wall tautology, NOT
publish-gating dominance; removable lean-consumer budget = the 2-7% serial bucket = ~5/10/23ms (grows with
chunk count, the DIS-24 mechanism) — REAL + faithful but BOUNDED, cannot close the ~22ms T16-Ephys gap
alone, ~0 at T7-T8. Dominant high-T binder = DECODE-WAIT (the gated asm engine; LEV-4 2.3x clean-rate) +
serial writev OUTPUT (DIS-5 surface). Owed-separately (NOT done): squishy cross-check (silesia = rg's
tuning corpus). Box restored cores:16/cpuset 0-15 (taskset16 fails, VERIFIED), thawed (no_turbo=0, wd
inactive), all-host pgrep-clean. OWES supervisor Opus gate. NEW tooling -> Steward: scripts/bench/
{_dis26_capture_guest.sh,_dis26_oracle_guest.sh} + worktree-only chunk_fetcher.rs:1665 delay knob. See
plans/disproof-ledger.md DIS-26.
