# ENGINE-W REFRESH (2026-06-12)

Base `bf4c65a8` (HEAD on branch `plans/engine-w-refresh`).
Charter predated by: asm rung-c DEFAULT-ON, keepIndex port, refillBuffer staging, mfast rejection,
rung-d marker-loop, storedheavy routing fix. This document refreshes the charter into the current
evidence state and proposes the actionable first increment.

---

## 1. DONE-vs-REMAINING (charter items, commit refs where applicable)

### SHIPPED — engine inner loop (P3.x + asm campaign)

| Item | Commit(s) | Outcome |
|------|-----------|---------|
| u8 flat-contig architecture (one-engine, flip-in-place) | Pre-exists; no separate commit — `decode_clean_into_contig` is the flat-u8 linear buffer in production (`marker_inflate.rs:2071`) | DONE; precondition for all asm work |
| P3.1 — DistTable single-lookup in contig fast loop | `ca241566` / `113116a3` | T1 -4%, DistTable differential live |
| P3.2 — runtime literal chaining (21.7M→10.9M lit iters) | `e63e0ccb` / `8af4fc29` | model T8 -6.8%, T1 -1.6% |
| P3.3 — decode-chain hardening | `b291bacd` | model +8.4%, T1 +2.4% |
| P3.4 — DistTable amortization + libdeflate copy + source prefetch | `a3401a58`, `181d7c25`, `02e6f962` | T1 silesia -87ms (1462→1375ms); T16 recovered |
| P3.5 — Rust decode-chain exhaustion pass (c1/c2/c4) | `8f93526b`, `5b85ffc5`, `1bfe0af3` | T1 +1.8%; "near exhaustion" of Rust scheduling headroom |
| Rung-a (litlen micro-asm) | `b5c3f7c4` (reverted, recoverable) | NO-SHIP: per-symbol seam tax ~1.4 cyc/crossing with zero latency recovered; mechanism banked |
| Rung-c — full-symbol-loop asm kernel, DEFAULT-ON | `90509c88`→`c7ba9578`→`e4609917`→`81e2366d` | native T1 -19.2% (939 vs 1170 ms), asm_frac 0.998; F2 spec-dist gather + F3 unconditional refill + D P3.4-shape copy all IN-KERNEL; DEFAULT-ON as of `e4609917` |
| Rung-d — marker-loop DistTable (N1) | `67431727` / `523ff9da` | TIE-KEEP; marker-loop wall-criticality CAUSALLY REFUTED on bignasa-isal T8 (2x slack under MFAST_DISABLE) |
| F2 speculative dist gather in-kernel | Incorporated in rung-c c3 (`90509c88`) | DONE (in-asm; exits to Rust only at subtable) |
| F3 unconditional branchless refill in-kernel | Incorporated in rung-c (VAR_VIII salvage §5 F3, asm-campaign.md) | DONE |
| D MOVDQU / P3.4-shape in-kernel copy | Incorporated in rung-c c3 | DONE (P3.4 shape transliterated; NOT the old VAR_VIII byte-copy-all) |
| VAR_VIII isolation bench | `922d6cbe` (bench/var8-fullkernel, parked) | SUPERSEDED by rung-c production kernel; isolation bench was stepping-stone, no longer the frontier |

### SHIPPED — pipeline economy (non-engine)

| Item | Commit(s) | Outcome |
|------|-----------|---------|
| keepIndex=false faithful port (window sparsity + zlib compression OFF) | `f60e8e3b` / `c31d9a07` | 430ms FINALIZE-SPARSITY CPU deleted; model-isal 1.19x → 1.13x |
| refillBuffer 128KiB staging port (last FFI-surface divergence) | `d1de7d48` / `a53c53e0` | TIE-KEEP |
| mfast throughput lever | Probe `10fda98b` / merge `6c1dc584` | REJECTED with mechanism: MFAST_DISABLE wall-flat (-0.6%) on bignasa-isal T8 and silesia-isal T4; ceiling ~0 |
| storedheavy routing fix (StoredParallel→ParallelSM demotion) | `36d04ca0` / `c78f98b1` | 1.53-1.60x LOSS → 0.50-0.65x WIN |
| Build flavor guards + refillBuffer demote counter/test | `f7d98a0e`, `32833f62` / `a53c53e0` | Live; fail-closed at script layer |

### REMAINING — engine-W track proper

| Item | Status | Evidence |
|------|--------|----------|
| Bootstrap/marker-prefix ~270ms (model-isal context) | NOT YET CLOSED | Direct rdtsc measurement: decodeBlock CPU = kernel 2,198 + FINALIZE-SPARSITY 430 [FIXED] + **bootstrap/prefix 270** + other ~110ms. The 270ms is the remaining NAMED delta after keepIndex fix. |
| Bootstrap-phase criticality perturbation at native T1 | NOT YET RUN | Process rule 1: must confirm bootstrap gates native T1 wall before work stretches. Native T1 silesia at HEAD = 939ms vs rg 810ms = 0.863x; how much of the 129ms gap is bootstrap is UNCHARTED. |
| Marker bootstrap asm kernel (rung-c sibling for the marker-decode loop) | NOT YET ATTEMPTED | Entry condition: bootstrap perturbation confirms criticality. Orchestrator (line 516): "the marker bootstrap loop is the UN-ASM'D sibling of the contig clean loop." |
| Bootstrap header/table-build economy vs rg's ~340 MB/s/thread | NOT YET MEASURED HEAD-to-HEAD | marker_inflate.rs:1546-1549 comment cites 340 MB/s target; bootstrap body measured at 171 MB/s (P3.4 era) — gap persists; whether it gates the wall is unconfirmed. |
| silesia T4/T16 isal near-bar cells | REBOOT-GATED | 0.860x / 0.992x; T16 already at bar; T4 is "drift-limited terminal" per the orchestrator; the gap is SCHEDULING EFFICIENCY (81% vs 92%), not engine symbol rate — separate track. |
| RSS column (~2.7x vs rg on model) | SEPARATELY OWNED | Not engine-W; deprioritized. |

---

## 2. WHERE THE REMAINING CYCLES ARE — cell × phase engine-criticality matrix

### Evidence basis

Two direct measurements constrain the matrix:

**A.** rdtsc combined-instrument ledger (orchestrator top entry): model-isal T8 decodeBlock CPU breakdown:
kernel 2,198ms + wrapper 42ms + CRC 35ms + boundary 33ms + FINALIZE-SPARSITY 430ms [FIXED by keepIndex] + **bootstrap/prefix 270ms** + ~10ms.
The 270ms is the thread-summed CPU across T8 workers on the model corpus (low compression ratio).

**B.** Phase decomp probe (orchestrator "PHASE DECOMP + REFRAME"): silesia T4 isal SUM 1763.7ms = header 19.5ms / marker body 901.6ms (proven-slack control) / ISA-L FFI 595.8ms / apply_window 91ms / residual 246.8ms. gzippy inflate-only SUM 1497ms vs rg 1781ms = 0.84x (gzippy does LESS inflate CPU); ideal wall 441 vs actual 543 = **81% efficiency vs rg ~92%**. The silesia T4 isal gap is SCHEDULING EFFICIENCY, not engine throughput.

**C.** Rung-c asm-campaign §10 frozen 3-way: native T1 silesia 939ms vs rg 810ms = 0.863x (gap 129ms). The asm kernel captured 35% of the decode ceiling (~225ms of ~620ms). Whether the remaining 129ms gap is bootstrap, scheduling, or output-path is UNCHARTED.

**D.** mfast DISABLE perturbation (orchestrator "PHASE-0 mfast PROBE"): MFAST_DISABLE (marker body → careful loop at 1.69x baseline decode cost) = wall-flat on both bignasa-isal T8 (-0.6%) and silesia-isal T4 (+0.4%). Marker body is causally SLACK for isal builds at T4+ on both corpora.

### Matrix

| Cell | Engine-critical? | What binds it | Engine phase that matters | Evidence cite |
|------|-----------------|---------------|--------------------------|---------------|
| native T1 silesia | **OPEN** — partial | Unknown; asm kernel covers clean-contig; bootstrap share uncharted | Bootstrap (window-absent prefix decode) is the unaddressed engine phase | asm-campaign §10 flip table; no bootstrap perturbation run at T1 |
| native T4 silesia | **OPEN** — partial | Mix of engine and scheduling at T4 | Bootstrap + scheduling | 0.860x; no T4 bootstrap perturbation |
| native T8 silesia | **PARTIAL** — shrinking | Marker + pipeline at T8; clean-contig only 136.8M of 211.9M bytes | asm kernel has ~3.8% effect at T8 (§9); residual is mostly non-engine | asm-campaign §9 T8 ON 331 vs OFF 347 = -4.6% |
| native T16 silesia | **PASS** | At bar (0.992x) | N/A | flip table |
| model-isal T8 | **YES — bootstrap 270ms** | 270ms bootstrap/prefix = the only remaining named delta | Bootstrap marker-decode (window-absent chunks, ~8.5ms/chunk × 51 chunks) | rdtsc combined instrument; orchestrator top entry |
| model-native all-T | **YES — engine-W dominant** | Low-ratio = large bootstrap fraction; clean-tail ~42% more CPU than rg's ISA-L (isal proxy) | Both bootstrap AND clean-contig at lower efficiency | orchestrator "POST-P0 REFRESH": native pure-Rust clean ~42% more CPU on model |
| weights-native | **YES** | Same mechanism as model; low-ratio | Bootstrap + clean-contig | Reported 1.14-1.70x loss; low-ratio corpus class |
| silesia-isal T4 | **NO** | Scheduling efficiency (81% vs 92%); marker body slack | None — ISA-L does the clean decode; marker body proven slack | phase decomp + mfast DISABLE flat |
| silesia-isal T8 | **NO / near-bar** | 0.973x; at-bar class | Marker body slack; scheduling marginal | flip table; mfast probe |
| bignasa-isal T8 | **NO — slack** | Marker body slack (mfast DISABLE -0.6%); 0.986x near-bar | None | mfast probe N=7; rung-d TIE-KEEP |
| bignasa-native T8 | **PARTIAL** | Mostly marker path; small gap (0.969x) | Marker bootstrap may contribute; low priority given small gap | flip table |
| storedheavy | **CLOSED** | Routing fix delivered 2-3x structural win | N/A | `c78f98b1` |

### Summary: engine symbol rate gates exactly

**T1 native (all corpora), model/weights at all T (native), model-isal T8 bootstrap phase.**
It is PROVEN SLACK for silesia-isal T4/T8, bignasa-isal T8 — the gap in those cells is parallel
scheduling efficiency and is a separate track. Do not route engine work at isal T4/T8 cells.

---

## 3. FIRST INCREMENT — proposal with pre-registered falsifier

### Recommendation: bootstrap-phase causal perturbation at native T1, then (if confirmed) marker bootstrap asm kernel

**Why this and not the alternatives:**

*F3 branchless refill in the existing kernel:* DONE — incorporated in rung-c (asm-campaign §5 F3
salvage). Not a remaining gap.

*F2 speculative dist gather in-kernel:* DONE — rung-c c3 includes the full dist decode path in-asm;
exits to Rust only at subtable/long-code (asm-campaign §2(c)).

*VAR_VIII isolation bench:* SUPERSEDED — rung-c is the production integration of that prior art.
No incremental value.

*Bootstrap header/table-build economy:* A valid second increment, but PRECEDED by the perturbation
— without knowing that bootstrap gates the native T1 wall, header economy work may be in slack.
`read_dynamic_huffman_coding` at ~340 MB/s target vs ~171 MB/s measured is a real gap;
it belongs in the increment ladder AFTER the perturbation confirms the phase is causal.

*silesia T4/T16 isal scheduling:* Different track, reboot-gated, NOT engine-W.

**The case for the bootstrap perturbation as increment 0:**

Process rule 1 (CLAUDE.md): "To test whether region R gates the wall, change R's time by a known
factor and measure the interleaved wall response." No such test has been run on the bootstrap
phase at native T1 post-asm-kernel. The 270ms model-isal rdtsc measurement is authoritative for
that context but does not directly prove native T1 silesia criticality. silesia is higher-ratio
than model → smaller bootstrap fraction. The 129ms native T1 gap may split between bootstrap and
other phases (scheduling, output path, or the asm kernel boundary residual). The perturbation
resolves this split in one frozen session.

**Pre-registered falsifier (increment 0 — perturbation):**

- *Instrument:* `GZIPPY_SLOW_BOOTSTRAP=N` — spin-inject N% of the bootstrap marker-decode phase
  (the `while !at_end_of_block` loop in the window-absent path of `gzip_chunk.rs`, the phase
  gating BEFORE the window flip to clean). Implement as a per-iteration spin proportional to
  measured loop iteration count (the `GZIPPY_SLOW_BOOTSTRAP` pattern analogous to
  `GZIPPY_SLOW_CLEAN_PCTL`).
- *Self-test:* binary-vs-itself at N=0: wall ratio must be 1.0 ± inter-run spread. Fail the
  instrument if this doesn't hold.
- *Frequency-neutral control:* run BOTH spin (SPIN=1) and sleep (SLEEP=1) variants at N=50 and
  N=100. If the delta survives sleep (core-yielding), criticality is real; if only spin shows a
  delta, it is a turbo artifact, not criticality.
- *Cells:* native T1 silesia, taskset -c 0, N≥9 interleaved, sha-verified both arms.
- *PASS (bootstrap critical):* +N% injection → +N% wall, proportional, sign-stable across reps,
  survives sleep-control. Then: enter marker bootstrap asm kernel work.
- *FAIL/SLACK (bootstrap not the binder):* wall flat (Δ < inter-run spread) under N=100 injection.
  Then: stop. Investigate the remaining 129ms gap via a different attribution (scheduling probe,
  output-path, or fresh removal oracle against HEAD).
- *N values to use:* N=50 (moderate) first; if proportional, confirm with N=100; if flat at N=50,
  also try N=100 as a final check before declaring slack. Total sessions: 1 frozen measurement
  with both N levels and sleep-vs-spin controls.

**Pre-registered falsifier (increment 1 — marker bootstrap asm kernel, entry conditional on above PASS):**

- *Construct:* a rung-c-parallel asm region for the marker-decode bootstrap loop
  (`read_internal_compressed_specialized::<true>` `'mfast` loop body when in window-absent mode).
  Same rung-c methodology: register-pinned bit-state + LUT base across the back-edge; exits to
  Rust at block boundaries / long-code / EOB; coverage counter (bootstrap_asm_frac ≥ 0.97 on
  model chunks is the validity gate). Use the VAR_VIII F1 litlen gather + F3 refill asm sequences
  (asm-campaign §5) — layout unchanged at HEAD.
- *Ship bar:* ≥2% of native T1 wall (≥18ms at 939ms base) on silesia. Additionally ≥2% of
  model-isal T8 wall if the 270ms bootstrap term is in scope (it should be: this kernel covers
  the same loop).
- *PASS gate (F-b1):* byte-exact sha grid (same gauntlet as rung-c §3) AND coverage ≥ 0.97 on
  model window-absent chunks AND frozen native T1 silesia −≥2% vs same-binary kill-switch.
- *KILL (TIE under valid coverage):* if the coverage gate holds (bootstrap_asm_frac ≥ 0.97) and
  T1 wall is flat — the bootstrap phase is already at the Rust scheduling plateau for ITS loop;
  no further asm headroom. Keep behind default-OFF gate (rule 7a), document TIE with coverage
  as the mechanism, close the bootstrap asm sub-track.
- *Bootstrap header/table-build economy:* evaluate as a THIRD increment regardless of whether
  increment 1 ships. `read_dynamic_huffman_coding` runs at ~171 MB/s vs rg's documented
  ~340 MB/s target; closing this gap is a Rust-level change (no asm required), independently
  landable, and combines additively with the asm kernel.

**Full gauntlet (per increment 1, if entered):**
1. Differential: bootstrap-asm path vs pure-Rust path over (i) model + silesia corpora
   (real-corpus rule: same commit), (ii) proptest random streams, (iii) adversarial fixtures
   (long-code, EOB at slop boundary, window-seeded mid-stream, stored-block boundaries).
2. Sha grid: {silesia, model} × T{1,8} × {asm-on, asm-off, base}.
3. fmt/clippy: default-features clean, gzippy-native warning count equals baseline.
4. Kill-switch: `GZIPPY_ASM_BOOTSTRAP=0` selects pure-Rust path (counter-verified, not assumed).
5. Knob exclusion: asm region auto-disables under contig_prof / slow_knob / removal-oracle env
   knobs (same charter §3.5 rule as for rung-c).
6. Coverage counter: bootstrap_asm_frac dumped in VERBOSE; failure to reach ≥ 0.97 on model
   window-absent chunks = instrument problem, not a valid rate result.

---

## 4. REBOOT-INDEPENDENT vs REBOOT-GATED

### Reboot-independent (T1 cells — safe to run now)

The "NEEDS NEUROTIC REBOOT" flag in the loss surface refers specifically to silesia T4/T16 isal
and bignasa T8 isal — near-bar cells where inter-session drift (rg absolutes shifting 506↔476ms,
gzippy 543↔556ms across sessions; sigma=40ms on silesia T4) makes the pass/fail verdict
unreliable. The drift source is uncontrolled box-state (L3 occupancy, turbo hysteresis, LXC
neighbor load) that a reboot and fresh freeze restores.

T1 measurements use `taskset -c 0` on one physical core. Inter-session drift at T1 is documented
as much smaller (sigma ~20ms at native T1 silesia in the asm-campaign flip gauntlet; rg T1
was stable at 810–816ms across N=11 + 3 confirm reps). A bootstrap perturbation at T1 involves
only that one core; the result is not confounded by L3 sharing or core-assignment drift.

**All T1 bootstrap work — perturbation experiment AND any subsequent marker bootstrap asm kernel
measurement at T1 — is reboot-independent.** The frozen interleaved protocol (bench-lock,
no_turbo=1, interleaved arms, N≥9) is sufficient for T1 without a reboot.

### Reboot-gated (T4+ near-bar cells — defer verdicts until reboot)

| Cell | Current ratio | Why reboot-gated |
|------|--------------|-----------------|
| silesia T4 isal | 0.860–0.912x (session band) | "drift-limited terminal"; sigma=40ms; worst-session bar binding; SCHEDULING gap not engine — separate track anyway |
| silesia T16 isal | 0.992x | Near bar; drift may flip verdict; confirm post-reboot |
| bignasa T8 isal | 0.975–0.986x (N=17 provisional) | Provisional pending N≥15 reconfirm; small gap, needs tight conditions |
| model T8 isal | 0.863x (new; 424ms vs rg 372-375ms) | Larger gap; 270ms bootstrap delta. The perturbation number itself (270ms direct rdtsc) is reboot-independent (it is a CPU-sum probe, not a wall ratio near bar). But CONFIRMING bar passage or ranking progress requires frozen conditions. |

Any work that ships primarily to improve T4+ near-bar cells should bank its frozen A/B AFTER the
neurotic reboot. Engine work motivated by T1 (the bootstrap perturbation + asm kernel) does not
need the reboot — it is measured at T1 and will generalize to T4+ as a consequence.

---

## 5. OPEN QUESTIONS NOT RESOLVED BY THIS REFRESH

1. What fraction of native T1 silesia's 129ms gap is bootstrap vs scheduling vs output path? (The
   perturbation answers this; it is OWED before any further engine speculation.)
2. Does the 270ms model-isal bootstrap consist entirely of the marker-decode loop, or does
   `read_dynamic_huffman_coding` (table-build economy) contribute a separable term? The rdtsc
   instrument labeled it "bootstrap/prefix" without decomposing into header-parse vs body-decode
   vs table-build. This decomposition belongs in the perturbation session's instrument design.
3. Post-reboot: are silesia T4 isal and bignasa T8 isal at-bar or still-LOSS? If still-LOSS,
   the scheduling/efficiency gap (81% vs 92%) needs its own track; that is NOT engine-W.
