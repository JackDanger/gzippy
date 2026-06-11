# ASM CAMPAIGN CHARTER — the decode-dependency-latency phase (engine/asm-p1)

Date: 2026-06-11. Base `777a8669` (post-P3.5). This charter opens the asm phase of
the funded engine campaign (plans/engine-campaign.md P3 endgame) on the complete,
gated data chain banked in plans/orchestrator-status.md (top entries) and
plans/removal-oracle-ceilings.md. It is incremental BY DESIGN — no big-bang port.
Every increment is separately landable, separately killable, byte-exact-gated, and
frozen-measured before the next is attempted.

## 0. Why asm, why now (the gated chain — do not re-litigate, re-verify)

1. **The decode chain owns ~half the wall** (removal oracles, disjointness-proven,
   conservative LOWER bound): NODECODE ceiling 642.6 ms = 50.9% of the native T1
   silesia wall (base 1263 ms at `8fa2042f`); 311.6 ms = 47.1% of masked model-T8
   (base 662 ms). The store side is bounded at 94 ms / 23 ms — exhausted by P3.4
   (contig-clean scope; gate-trim #2 applies).
2. **The instructions are already right; the LATENCY is wrong.** Native builds
   already emit BMI2 (disasm-proven, P3.4 NO-SHIP of the dispatch lever). The cost
   is the dependent-load chain per symbol: litlen LUT load → bit-extract/consume →
   branch tree → dist LUT load → extra-bits extract → copy-source load.
3. **Rust-level rescheduling is near exhaustion.** P3.5 (c1 spec-consume hoist, c2
   fused litlen→dist spec load, c4 prefilled decode) extracted +1.8% of wall
   (~3.3% of the decode ceiling); c3 was refuted with hardware mispredict data.
   T1 frozen 1183 → 1162. The remaining ~620 ms-class chain does not move from
   inside Rust's scheduling model.
4. **The asm existence proof is banked**: VAR_VIII (full-kernel `core::arch::asm!`,
   register-pinned, byte-exact, asm_frac 0.938) measured **+14.6% over LLVM's best
   loop** (sign-stable on all 5 chunks, copy-confound disproved first-hand —
   plans/var8-gate-verdict.md, result SOUND) on the OLD pre-one-engine
   architecture. The campaign ruled prior falsifications non-binding; the +14.6%
   is prior ART, not a prior ceiling — it must be re-based against the P3.5 loop.

Scope of the target: the contig clean fast loop's per-symbol decode chain —
`Block::decode_clean_into_contig` (`src/decompress/parallel/marker_inflate.rs:2186`),
fast loop `:2466-2760` — specifically litlen lookup + consume + dispatch, the
dist lookup/consume, and (only at the last rung) the loop back-edge itself.
Vendor reference shape: `vendor/isa-l/igzip/igzip_decode_block_stateless.asm`
(top guard :508-512, F1 litlen gather :524-540, F2 speculative dist gather
:550-552, F3 unconditional refill :528-547, D copy :591-627).

## 1. Budget math (ceilings per cell — what a perfect outcome buys)

| cell | wall (frozen) | decode ceiling | rg wall | gap to rg | gap / ceiling |
|------|---------------|----------------|---------|-----------|---------------|
| native silesia T1 | 1162 ms (post-P3.5; ceiling measured at 1263 base) | 642.6 ms (50.9%) | ~926.6 ms | ~235 ms | ~37% |
| native model T8 (masked) | 662 ms | 311.6 ms (47.1%) | — (internal ratio; gate-trim #1) | — | — |

- The ENTIRE T1 rg gap (~235 ms) fits inside the decode ceiling ~2.7× over —
  removing decode compute lands at 620 ms, far past rg. Headroom is real and
  lives exactly where the asm work goes (removal-oracle verdict).
- **Sanity bound from prior art**: VAR_VIII's +14.6% over LLVM-best applied to a
  ~620 ms-class chain ≈ 80-90 ms-class IF the new loop still sits at the old
  LLVM plateau. P3.1-P3.5 already banked some of that headroom (local-Bits
  mirror ≈ register pinning lite; spec_dist ≈ F2 lite), so the realizable asm
  delta is SMALLER than 14.6% and must be re-measured, not assumed.
- **Ship bar (campaign rule)**: ≥2% of T1 wall (≥ ~23 ms) per increment =
  ship-class. Δ < spread = TIE; a byte-exact TIE is kept/layered (rule 7a) but
  never claimed as a win. A NO-SHIP with a latency analysis is an acceptable
  increment outcome.
- **Kill line for the whole phase**: if the FULL kernel (rung (c)) ties best-Rust
  within spread under a valid coverage counter, the in-process plateau is
  EARNED — engine-W is closed as unreachable in-process and the campaign
  conclusion gate fires. That is a legitimate, fundable outcome.

## 2. Increment ladder (smallest landable asm units first)

Rungs are strictly ordered; each rung's verdict (ship / TIE-keep / no-ship +
mechanism) is banked before the next starts. Rationale: rungs (a)/(b) are cheap
falsifiers that also build the permanent scaffolding (feature gate, dispatch,
differential harness, decide.sh knob row) the full kernel needs anyway.

### (a) litlen decode+consume micro-sequence as an `asm!` block in the existing Rust loop
- Replace the hot litlen lookup+consume sequence (`LutLitLenCode` short-table
  gather + `Bits::consume`) with one `asm!` block: `mov`-load
  `short_tbl[bitbuf & 0xFFF]`, LARGE_FLAG/zero-len test (flag out for the Rust
  long path), `shrx` consume, `sub` bitsleft — VAR_VIII lines :1704-1715
  salvaged near-verbatim (layout unchanged at HEAD, §5). Placement: the chain
  arm's `decode` (`marker_inflate.rs:2570`) and/or the bottom preload — exact
  site pre-registered in the increment's falsifier doc before measuring.
- **Hypothesis under test**: LLVM emits the right instructions but schedules the
  chain suboptimally around the branch tree. Expected outcome honestly stated:
  TIE-to-small — the same 5 instructions in the same dependence chain have the
  same latency; an `asm!` block is also an LLVM scheduling BARRIER and can
  plausibly LOSE by breaking P3.5's c2/c4 placements. (a) is worth its cost
  because (i) it is the cheapest disproof of "hand-scheduling helps at micro
  scale", (ii) a LOSS localizes the boundary tax that rung (c) must amortize,
  and (iii) it lands the scaffolding.
- Ship-or-no-ship: frozen masked T1 A/B, n≥9, byte-exact grid first.

### (b) fused litlen→dist pair (the backref-arm latency chain) in one `asm!` block
- The lone-length arm end-to-end: trailing-code classify → dist
  `DistTable::lookup(bitbuf)` issued SPECULATIVELY before the EOB/MAX/length
  branch tree resolves (full igzip F2; P3.5 c2's `spec_dist` got the load
  hoisted but the consume+decode_distance chain still serializes behind the
  branch tree) → `consume_entry` → `decode_distance`, with the subtable branch
  flagged out to Rust. This is the longest serial chain in the loop (backref
  62.6% of classed cycles, contig_prof @ T8).
- NOTE the dist side is the NEW (libdeflate-shape) table — `DistEntry` u32,
  TABLE_BITS=9, premultiplied entry decode — NOT VAR_VIII's ISA-L small LUT;
  the asm here is a REWRITE (shorter: one load + one bzhi/shrx-class decode vs
  igzip's three loads; see salvage verdict §5).
- Ship-or-no-ship: same protocol as (a). A (b) win with an (a) TIE is the
  expected signature of "latency lives in the cross-branch dist chain".

### (c) the full symbol loop in asm (VAR_VIII rebuilt on the P3.5 contract) — ONLY if (a)/(b) prove or precisely localize the latency win
- The construct prior art proved +14.6% in isolation: back-edge inside the asm,
  bit-state (`bitbuf`/`bitsleft`/`pos`) + out cursor + litlen table base pinned
  across iterations (igzip F4), unconditional refill (F3), full F2, the P3.4
  copy shape transliterated (NOT VAR_VIII's byte-copy-all-overlaps D loop),
  exits to Rust only at long-code / EOB / slop boundary / invalid — each exit
  leaving the cursor before a fresh un-consumed symbol (the VAR_VII/VIII
  invariant that makes the seam byte-exact).
- Must carry the P3.2 lit-chain arm and the lone-literal 1-byte store (advisor
  Q3) — the kernel's old unconditional 8-byte store on lone literals is
  pre-Q3 and refuted.
- Coverage counter (`asm_frac ≥ ~0.97` on clean silesia chunks) is a VALIDITY
  precondition for any rate claim (the DIS-1 confound guard).
- Entry condition for building (c): (a)/(b) banked verdicts that either show a
  win (scale it) or show TIEs whose mechanism is the asm-block boundary tax
  (then (c) is exactly the construct that amortizes the boundary to
  ~once-per-block). If (a)/(b) instead show "no latency recoverable even
  without boundaries" — e.g. (b) ties AND its perf-counter profile shows the
  dist chain already fully overlapped — then (c) is NOT entered and the phase
  concludes with mechanism (rule 7a satisfied: that IS a mechanism, not a
  narrow miss).

## 3. Verification gauntlet (per increment — no exceptions)

1. **Differential, the lut-differential precedent**: asm path vs pure-Rust path
   on identical inputs must produce identical output bytes AND identical bit
   cursor (`pos`/`bitbuf`/`bitsleft`) at every exit, over (i) the silesia +
   model corpus (real-corpus rule: ships in the same commit), (ii) proptest
   random streams, (iii) adversarial fixtures: long litlen codes, long/subtable
   dist codes, EOB at slop boundary, invalid codes, dist==1 RLE, dist>length
   overlap, near-end-of-input fabricated-bits region.
2. **Byte-exact sha grid**: {silesia, model} × T{1,8}, gzippy-native, frozen
   guest, vs pinned shas. Plus the flip-seam adversarial test
   (`contig_clean_matches_ring_clean_on_*`) and the full lib suite, BOTH
   feature sets (default AND `--no-default-features --features
   pure-rust-inflate[,asm-kernel]`), local Rosetta x86-64-v2 + guest native.
3. **fmt/clippy** default-features clean; gzippy-native warning count ==
   baseline.
4. **Kill-switch proof**: `GZIPPY_ASM_KERNEL=0` produces the pure-Rust path
   (counter-verified, not assumed), same binary — the layout-phantom
   discriminator every campaign A/B relies on.
5. **Measurement-knob exclusion**: the asm path auto-disables when
   `contig_prof` / `slow_knob` / removal-oracle env knobs are active (hooks
   cannot fire inside asm; a silently-unperturbable region would poison every
   future causal instrument). Asserted by a test.

## 4. Kill-switch / feature-gate policy

- Cargo feature `asm-kernel`, compiled only on `target_arch = "x86_64"`;
  the pure-Rust loop is ALWAYS compiled and always reachable (runtime
  dispatch). Non-x86_64 builds are bit-for-bit unaffected.
- Runtime kill: `GZIPPY_ASM_KERNEL=0` (OnceLock env read, one predictable
  branch at contig-call entry — the slow_knob pattern).
- Default-OFF until the increment's ship gate passes frozen; flipping the
  default is its own reviewed commit.
- The asm and the Rust loop are bound by the differential suite FOREVER — any
  future loop change must keep both green or remove the asm (no drift).

## 5. VAR_VIII SALVAGE VERDICT (the banked prior art, evaluated against the P3.1-P3.5 loop)

**Located**: the prototype was never committed anywhere — it existed only as
uncommitted modifications in the stale worktree
`.claude/worktrees/var8-fullkernel/benches/engine_isolation.rs:1554-1920`
(branch parked at `d56cb0f5`). Now BANKED as `bench/var8-fullkernel` @
`922d6cbe` (this session). Gate verdict plans/var8-gate-verdict.md: result
SOUND (+14.6% over VAR_VI, 0.667× ISA-L, byte-exact, asm_frac 0.938), HOLD on
integration — measured on the OLD architecture, ruled non-binding.

**SALVAGE (reusable against the new loop, with mechanism):**
1. *The F1 litlen gather+consume asm sequence* (`:1704-1727`): the
   `LutLitLenCode` layout is UNCHANGED at HEAD (`lut_huffman.rs:61-75,243` —
   `short_code_lookup: [u32; 1<<12]`, LARGE_FLAG_BIT=1<<25, sym mask 0x1FFFFFF,
   count bits 26-27, len bits 28-31), so the gather/flag-test/`shrx`-consume
   block drops into rung (a) near-verbatim.
2. *The F3 unconditional-refill asm sequence* (`:1694-1703`): the `Bits`
   convention is UNCHANGED (`consume_first_decode.rs:245-265` — `bitsleft|56`,
   shift-consume; the asm's `(63-ril)>>3` byte-advance ≡ the Rust
   `7-((bits>>3)&7)`). Valid for rung (c); rungs (a)/(b) keep Rust refills.
3. *The exit-code protocol + invariant*: speculative `mov {ret}` exit codes,
   single end-label, and the "every exit leaves the cursor before a fresh
   un-consumed symbol" contract — the property that made the seam byte-exact.
   Reuse wholesale in (c).
4. *The KernCtx cold-spill pattern* (`:1579-1586`): loop-invariants via
   `[ctx+off]` memory operands keeps register pressure at 12 operands. Reuse
   in (c) (membership changes: dist table base, out_limit/in_limit recomputed
   for the contig slop scheme).
5. *The instrument*: VIII_COVERAGE asm_bytes/tail_bytes/reentries counters and
   the pre-registered gate STRUCTURE (byte-exact AND coverage AND rate vs
   pre-named comparators, KILL = plateau-vs-LLVM). Reuse with new comparators:
   the PRODUCTION P3.5 loop replaces VAR_VI as "LLVM's best".

**REWRITE (diverged or refuted since the prototype):**
1. *The F2 dist path*: kernel gathers the ISA-L small LUT (u16[1<<10],
   SMALL_FLAG_BIT, then DISTANCE_BASE + DISTANCE_EXTRA loads). Production now
   uses the libdeflate-shape `DistTable` (`libdeflate_entry.rs:236,564` —
   `DistEntry` u32, TABLE_BITS=9, subtable ptrs, `consume_entry` +
   `decode_distance(saved_bitbuf)`). Rewrite the asm against `DistEntry`
   (one load + in-register entry decode — SHORTER chain than igzip's three
   loads; the subtable case exits to Rust like the long-code case).
2. *The D backref copy*: kernel byte-copies ALL overlapping cases (incl.
   dist==1) and ymm-copies non-overlap. P3.4's `emit_backref_contig` shape
   (dist≥8 burst+stride word copy, dist==1 RLE fill, overlap-correct,
   envelope-proven 264<266) is measured better (-87 ms T1) — transliterate
   THAT shape into the (c) kernel, do not salvage the kernel's copy.
3. *The loop-arm structure*: add the P3.2 runtime lit-chain (≤2 extras, the
   ~1.96-2.57 lits/iter mechanism) and the lone-literal 1-byte store (advisor
   Q3 — the kernel's unconditional 8-byte store on sym_count==1 wastes
   bandwidth); keep the packed-u64 store only for sym_count>1.
4. *The integration seams*: production's commit/oracle/prof/slow-knob hooks
   and the careful-loop handoff (`FAST_OUT_SLOP`/`FAST_IN_SLOP` scheme)
   replace the bench driver's target_end scheme; §3.5 knob exclusion applies.

**DISCARD**: the bench-only block driver (`decode_var_viii` outer loop,
uncompressed-block arm, `build_block_tables`, Vec-returning tails) — production
owns all of that; and the `options(nostack)` setting if callee-saved pressure
appears in (c) (igzip itself uses a stack frame; feasibility §3.1).

**Net**: the hard, risky 40% of a full kernel (bit-engine mirror, litlen
gather, exit protocol, coverage instrument, gate design) is salvaged and
already byte-exact-proven once; the dist path, copy, and arm structure are
rewritten to the P3.4/P3.5 shapes they must be byte-identical to anyway.

## 6. Measurement protocol (every number in this phase)

- Frozen host: `bench-lock` acquire + no_turbo=1/governor readback on the guest
  (run under bash — the zsh `$SSH_JUMP` word-split trap is documented in
  plans/removal-oracle-ceilings.md); release + RESTORE verified after.
- Canonical mask: T1 `taskset -c 0`; T8 `taskset -c 0,2,4,6,8,10,12,14`.
- `GZIPPY_FORCE_PARALLEL_SM=1`, `GZIPPY_DEBUG=1` path assertion
  (`path=ParallelSM`), build `--no-default-features --features
  pure-rust-inflate[,asm-kernel]`, target-cpu native on guest.
- Interleaved best-of-N≥9 (N=21 for close calls); sha-verified EVERY arm; both
  same-binary (kill-switch) AND cross-binary A/Bs — same-binary is the causal
  currency, cross-binary catches layout phantoms.
- decide.sh integration: add a `GZIPPY_ASM_KERNEL` knob row so `fulcrum
  decide` / decide.sh runs carry the asm A/B permanently (EFFECT-VERIFIED
  predicate = the dispatch counter, not line presence).
- Disk check before every guest build (`df -h`) — full-disk trap is documented.
- Numbers from the standing instruments, never hand-rolled one-off scripts.

## 7. Pre-registered phase falsifiers (stated before any asm is written)

- F-a: rung (a) interleaved frozen T1 delta vs same-binary kill-switch OFF.
  Ship ≥2%; TIE → keep-if-byte-exact, bank "micro-asm scheduling is not the
  lever" with the perf profile as mechanism.
- F-b: rung (b) same protocol. Ship ≥2%. TIE + overlap-evidence → the dist
  chain is already absorbed; (c) entry DENIED unless boundary-tax evidence
  says otherwise.
- F-c: rung (c) PASS = byte-exact AND asm_frac ≥0.97 AND frozen T1 wall
  −≥2%; KILL = tie vs P3.5 loop within spread under valid coverage ⇒
  in-process plateau EARNED ⇒ phase ends, conclusion gate fires.
- Any increment that is byte-exact and TIEs is KEPT behind its default-OFF
  gate (rule 7a) — the falsification record stays honest either way.

---

## 8. F-a VERDICT (2026-06-11, same session): NO-SHIP — the per-symbol asm boundary tax is real and quantified

Increment (a) built, differential-gated, and frozen-measured (commit
`b5c3f7c4`). **NO-SHIP**: the asm micro-step makes the wall WORSE.

- **Correctness (all green):** step differential asm==ref over 400k random
  states × {fixed, skewed} tables ON REAL BMI2 (guest; local Rosetta exposes
  no BMI2 — the asm half only executes on the box); sha grid 12/12
  byte-identical across {silesia, model} × T{1,8} × {on, off, base}, silesia
  == the pinned corpus sha; kill-switch proven (enabled=false, 0 hits);
  effect counters prove engagement (T1 silesia 10.89M hits / 5.05M bails =
  68% of chain candidates consumed in asm).
- **Frozen interleaved A/B** (bench-lock acquired, no_turbo=1 readback,
  RESTORE VERIFIED after; arms on/off/base interleaved, n=9, sink /dev/null;
  display truncation lost T1 reps 1-3 — medians below are over the captured
  reps, sign-stable on every captured rep):
  - T1 silesia (taskset -c 0): ON ~1214-1217 ms vs same-binary OFF
    ~1197-1198 ms → **ON +16 ms (-1.3%)**; cross-binary base 1181-1186 ms.
  - T8 model (masked): ON ~671 vs OFF ~656.5 vs base ~658 → **ON +14 ms
    (-2.2%)**; OFF≈base TIE.
- **Mechanism (the latency analysis the campaign rules require):** ~15.9M
  step invocations at T1 silesia carry the asm-block boundary each —
  operand setup (bitbuf/bitsleft copied to fresh locals + written back), an
  LLVM scheduling barrier inside the chain loop (breaks the P3.5 c2/c4
  placements' freedom), and a duplicated short-LUT probe on every bail
  (32%). +16 ms / 15.9M ≈ 1.0 ns ≈ **~1.4 cycles of tax per step at the
  frozen 1.4 GHz** — with ZERO latency recovered, because the same
  load→extract→branch chain executes either way. The hand-written
  instructions are not better than LLVM's; the per-symbol seam is pure cost.
- **Banked finding (F-a, pre-registered):** micro-asm scheduling is NOT the
  lever. This is the DIS-1 lesson re-confirmed on the NEW loop with a
  10×-finer instrument: the asm seam costs ~1-2 cycles per crossing, so any
  winning asm construct must amortize the boundary to ~once-per-block —
  exactly rung (c)'s back-edge-inside-asm shape (VAR_VIII's +14.6% was
  measured with ~3.4-4.3K reentries/chunk vs the ~15.9M here, a ~4000×
  difference in crossings).
- **Disposition (rule 7a):** byte-exact code KEPT behind the default-OFF
  `asm-kernel` feature (release builds unaffected — base binary measured
  identical-class to P3.5); the scaffolding (feature gate, kill-switch,
  effect counters, differential harness, guest A/B driver) is the permanent
  asset rungs (b)/(c) reuse. Rung (b) remains worth one attempt ONLY in a
  shape that does not add a per-symbol seam to the lit path (the dist-side
  fusion fires ~once per backref, ~5× rarer than chain steps); otherwise
  proceed directly to the rung-(c) entry decision with F-a as evidence that
  the boundary, not the headroom, killed (a).


---

## INCREMENT (a) FALSIFIED — banked record (2026-06-11)

Built byte-exact (commit b5c3f7c4, REVERTED from the branch per the no-dead-code
rule; recoverable from history). Frozen A/B: +16ms T1 / +14ms model-T8 — a
REGRESSION. MECHANISM: the Rust<->asm! boundary seam tax (~1.4 cyc/crossing x
15.9M crossings) exceeds any micro-step latency gain; LLVM cannot schedule
across the asm! boundary, so the seam re-serializes the very chain the kernel
tried to shorten. This is VAR_VIII's re-entry-spill lesson at micro scale and
CONFIRMS the ladder's design premise: the asm must OWN a region large enough
to amortize crossings — rung (c) (the full symbol loop, one entry/exit per
fast-loop run) is the only shape with a chance. Rung (b) (fused pair) is
likely seam-bound too; evaluate analytically before building.
NEXT: rung (c) per the charter — full-symbol-loop asm with the VAR_VIII
salvage (register contract rewritten for the P3.1-P3.5 loop state), the
~620ms budget, and the seam amortized to ~once per chunk.
