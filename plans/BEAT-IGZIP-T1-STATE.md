# BEAT-IGZIP-T1 — DURABLE STATE

## ====== LOCALIZE SESSION (2026-06-18 PM) — WHERE IS THE PER-ITERATION GAP? ======
GOAL: decompose gzippy run_contig vs igzip decode_huffman kernel into per-region
cyc/B (refill / copy / classify-decode / loop-overhead), confirm top candidate
causally. Intel cpu4 unstressed, /dev/null sink, freq-invariant cyc/B. SINGLE-ARCH
= NOT LAW (AMD owed). Tools committed: scripts/bench/_localize_profile_guest.sh
(perf record+annotate both kernels), scripts/bench/_localize_bucket.py (deterministic
address→region bucketer, SELF-VALIDATES sum=100%), scripts/bench/_localize_patch_perturb.py.

### GATE-0 CATCH (instrument self-validation — the most-violated rule, caught here)
The prior /tmp/symtarget was STALE: built @ 2b10aa48 (PRE-flagbit); its disasm showed
the OLD shrx→cmp$0xff literal discriminator, NOT the production flag-bit
`test 0x1000000;jnz 49f`. REBUILT symtarget from mission tip 0a592a54
(CARGO_TARGET_DIR=/tmp/symtarget on tmpfs; strip=false debug=2 target-cpu=native;
asm pulled by pure-rust-inflate). New disasm has BOTH `test 0x2000000`(long) AND
`test 0x1000000`(flag-bit), cmp$0xff now COLD. SHA==zcat (silesia+nasa), entries=24128.
REBUILD: `cd /root/gz-fullrewrite && git checkout 0a592a54 && RUSTFLAGS="-C target-cpu=native"
CARGO_TARGET_DIR=/tmp/symtarget CARGO_PROFILE_RELEASE_STRIP=false CARGO_PROFILE_RELEASE_DEBUG=2
cargo build --release --no-default-features --features pure-rust-inflate`

### DELIVERABLE 1 — per-region cyc/B, KERNEL-to-KERNEL (perf -F5000, 12 loops, self-validated sum=100%)
SILESIA (gz kernel 4.889 = 82.95%×total/B; ig kernel 3.981 = 89.14%×; KERNEL gap +0.908):
| region                          | gz %  | ig %  | gz c/B | ig c/B | gz−ig  |
|---------------------------------|-------|-------|--------|--------|--------|
| bit-mgmt (consume+refill)       | 19.1  | 14.4  | 0.934  | 0.573  | +0.361 |
| backref-copy (MOVDQU)           | 19.1  | 28.0  | 0.932  | 1.115  | −0.183 |
| classify/decode/table/EOB       | 46.9  | 47.8  | 2.294  | 1.905  | +0.389 |
| loop-overhead (guard/store/spill)| 14.9 | 9.8   | 0.730  | 0.389  | +0.341 |
NASA (gz kernel 1.843=54.22%×; ig 1.390=87.73%×; KERNEL gap +0.453):
| bit-mgmt   16.2/13.9  0.299/0.193 +0.105 | copy 26.5/35.7 0.488/0.496 −0.008 |
| classify   39.4/39.2  0.726/0.545 +0.180 | loop-ovh 18.0/11.2 0.331/0.155 +0.176 |
⇒ KERNEL gap is **DIFFUSE** — spread ~evenly across bit-mgmt + classify + loop-overhead;
  **backref-copy is at PARITY-or-BETTER** (gzippy's MOVDQU port already matches igzip;
  do NOT attack copy). NO single dominant kernel sub-lever.

### DELIVERABLE 1b — WHOLE-PROCESS gap ≫ KERNEL gap: the gap is SCAFFOLD-DOMINATED
| corpus  | whole gz | whole ig | whole gap | kernel gap | SCAFFOLD gap | scaffold %of gap |
|---------|----------|----------|-----------|------------|--------------|------------------|
| silesia | 5.894    | 4.466    | +1.428    | +0.908     | **+0.520**   | 36%              |
| nasa    | 3.399    | 1.584    | +1.815    | +0.453     | **+1.362**   | **75%**          |
gz scaffold (nasa, %proc→cyc/B): asm_exc_page_fault 9.32%(0.317) + __memmove_avx 7.35%
(0.250) + finish_decode 4.03%(0.137) + sync_regs 2.48%(0.084) + clear_page 1.71%(0.058).
ig scaffold (nasa): crc32 7.82%(0.124) + copy_to_iter 1.39%. ⇒ On backref-heavy data the
bulk of the igzip deficit is OUTPUT memmove + first-touch PAGE FAULTS + Rust bail-glue —
i.e. the rapidgzip-ARCHITECTURE/consumer domain, NOT the inner-loop machinery.
(HYPOTHESIS-tier: perf attribution; scaffold NOT yet causally perturbed — see NEXT.)

### DELIVERABLE 2 — Gate-2 CAUSAL perturbation of the REFILL/bit-mgmt chain (named top suspect)
THROWAWAY K8 build (mission tip + 4 rol/ror pairs = 8 value-NEUTRAL serially-DEPENDENT
ops on {bitbuf} right after the hot `6:` refill `or {bitbuf},{t2}`, extending the
bitbuf→preload/consume recurrence). CONTROLS: K0 sha==new-native 891c9925 (trusted
baseline); K8 sha 82f1eeab DISTINCT + byte-EXACT silesia/nasa + entries=24128 +
16 extra rol/ror in .text (asm! opaque to optimizer, non-inert PASS); GHz spread PASS;
self-test (A2-A1) PASS. N=21 paired, unstressed, /dev/null. medΔ=(K8−K0):
| corpus  | medΔ cyc/B | per-op    | 95%CI            | Wilcox p  | ΔIPC   | verdict        |
|---------|------------|-----------|------------------|-----------|--------|----------------|
| silesia | +1.007 (+17.0%) | +0.126 | [+0.989,+1.020]  | 6.412e-05 | −0.205 | **ON CRITICAL PATH** |
| nasa    | +0.133 (+3.9%)  | +0.017 | [+0.086,+0.170]  | 0.0136    | −0.028 | WASH (fails p<0.01) |
⇒ silesia: PROPORTIONAL, large, p<0.01, ΔIPC dropped (latency-bound) ⇒ the refill/bit-mgmt
  chain IS on the critical recurrence — shortening it CAN move cyc/B. nasa: weak/wash
  (nasa's hot literal refill runs rarely; its backref X1 refill — unpatched — is the
  relevant one). COMBINED w/ STATE's banked classify-chain perturbation (proportional),
  the top-2 kernel candidates (refill + classify) are BOTH causally on-path.
  Loop-overhead (+0.341 sil) NOT perturbed this turn (attribution-only).
  RE-VERIFY: rebuild K0/K8 (`/tmp/loc_prof/patch_perturb.py <asm> 4`, prod flags), then
  `BIN_A=/tmp/bin/gzippy-k0 BIN_B=/tmp/bin/gzippy-k8 PIN=4 REPS=21 CORPORA="silesia nasa"
  SKIP_STRESS=1 GZIPPY_FORCE_PARALLEL_SM=1 bash /root/distpreload-harness/_distpreload_paired_guest.sh`

### DELIVERABLE 3 — removal-oracle BOUNDS (ceiling ≠ gain)
KERNEL machinery ceiling (igzip-parity removal-oracle, perf-decomposed): if gzippy's
kernel matched igzip per-region → recover +0.908 cyc/B (sil) / +0.453 (nasa) MAX —
diffuse, so realistically only fractions capturable. Per-region ceilings (sil):
refill +0.361, classify +0.389, loop-ovh +0.341, copy 0 (already ≤). SCAFFOLD ceiling
(whole−kernel): +0.520 (sil) / +1.362 (nasa) — LARGER, esp. backref-heavy.

### RECOMMENDATION (gated-HYPOTHESIS; Intel-only NOT-YET-LAW; for next-technique decision)
1. **Do NOT chase a single kernel micro-lever.** The kernel per-iteration gap is small
   (+0.91 sil/+0.45 nasa) and DIFFUSE; copy is already at parity. The mission premise
   ("igzip's leaner per-iteration machinery") is CONFIRMED real (refill on-path) but
   bounded and spread — no peephole captures it; matching igzip needs its whole fused
   shape (consume+refill+preload interleave), high-effort/low-marginal.
2. **The biggest causally-confirmed KERNEL sub-lever = refill/bit-mgmt** (silesia, Gate-2
   on-path, ceiling +0.36 cyc/B). Byte-exact technique to target it: fuse gzippy's
   consume (`shrx;sub`) + refill (`mov;shlx;or;63-sub-shr3-add;or56`) + next-litlen
   preload into igzip's single interleaved cadence (igzip 38d0e-38d71 does consume,
   speculative store, ONE refill-advance, and BOTH next-litlen+dist preloads in one
   straight chain with the load-use hidden) — i.e. reduce the consume+refill DEPENDENT
   chain LENGTH, not the instruction count. Pre-register success = silesia cyc/B medΔ<0 p<0.01.
3. **HIGHER-VoI but ARCHITECTURE-domain: the SCAFFOLD** (whole gap ≫ kernel gap; nasa
   75%). Next causal test (owed, not done): removal-oracle on output __memmove_avx +
   first-touch page faults — decode into a REUSED/pre-faulted output buffer (no per-chunk
   fresh alloc) and/or eliminate the libc memmove copy; measure nasa whole-process cyc/B.
   If it drops proportionally, the consumer/chunk-output path (faithful rapidgzip port
   territory) is the real remaining lever on backref-heavy corpora — ESCALATE to user (R3:
   inner-loop is near its floor; the gap is moving to architecture).
4. AMD/Zen2 replication of all of the above is owed before LAW.
GUEST ARTIFACTS kept: /tmp/symtarget (symboled mission-tip bin), /tmp/bin/gzippy-k0
(=new-native), /tmp/bin/gzippy-k8 (perturb), /tmp/loc_prof/ (annot+stat+bucket.py).
gz-fullrewrite moved to 0a592a54 (was 2b10aa48); asm reverted clean; box: powersave,
0 stressors, no pinning leftover.

## ====== B-SIZING SESSION (2026-06-18) — 3 GATING MEASUREMENTS BEFORE FUNDING B ======
Goal: size rewrite B (igzip-style fat DIRECT one-symbol-per-iter table, ~0 unpack)
with MEASURED numbers (not the 1.34 slope/ceiling) for a USER R3 decision. Guest
trainer cpu4, unstressed, /dev/null sink. Binaries: packON=/root/bin/gzippy-new-native
(=mission tip 76c750f8; rebuild gzippy-packON sha-IDENTICAL → provenance proven),
packOFF=/root/bin/gzippy-packOFF (SINGLE_SYM_FLAG @ lut_huffman.rs:1041; byte-exact
silesia/nasa/monorepo; source REVERTED to TRIPLE).

### MEASUREMENT 1 — REAL-regime memory-stall fraction (is the 1.34 headroom real or memory-capped?)
new-native, unstressed, cpu4, perf -r 9. ANSWER: **COMPUTE-BOUND = YES.**
| corpus  | cycles    | L1d-miss-stall | L2-miss-stall | L3-miss(DRAM)-stall | TMA mem-bound | TMA core-bound |
|---------|-----------|----------------|---------------|---------------------|---------------|----------------|
| silesia | 1.244e9   | 2.39%          | 1.17%         | **0.69%**           | 7.0%          | 15.2%          |
| nasa    | 0.686e9   | 6.06%          | 3.98%         | **2.73%**           | 16.7%         | 16.0%          |
⇒ Tiny DRAM-stall fraction (silesia 0.69%, nasa 2.73% of cycles). TMA agrees: silesia
mem-bound only 7.0% (core-bound 15.2%); nasa mem-bound 16.7% (core-bound 16.0%). The
unstressed (production) regime is COMPUTE-bound — esp. silesia where B's payoff is
largest. ⇒ B's instruction-reduction headroom (the ~1.34 cyc/B oracle) is REAL in
production, NOT memory-capped. The earlier stressor-WASH was an artifact of the
artificial 14-thread bandwidth-saturation, not the real regime.
RE-VERIFY: `taskset -c 4 perf stat -r 9 -e cpu_core/cycles/,cpu_core/instructions/,cpu_core/memory_activity.stalls_l1d_miss/,cpu_core/memory_activity.stalls_l2_miss/,cpu_core/memory_activity.stalls_l3_miss/ -- env GZIPPY_FORCE_PARALLEL_SM=1 taskset -c 4 /root/bin/gzippy-new-native -d -c -p1 /root/silesia.gz >/dev/null`

### MEASUREMENT 2 — what does multi-symbol packing (which B ABANDONS) currently buy?
TOGGLE FOUND: lut_huffman.rs:1041 TRIPLE_SYM_FLAG→SINGLE_SYM_FLAG forces 1-sym/iter
through the EXISTING table (still pays unpack). packON(A=new-native) vs packOFF(B),
N=21 paired, unstressed, all SIGNIF (p=6.4e-5, CI excl 0), self-tests PASS, non-inert
(packOFF Δinstr/B>0 = more loop iters). medΔ=(packOFF−packON) = **packing_benefit =
what B costs us:**
| corpus  | packON cyc/B | packOFF cyc/B | packing_benefit medΔ | Δinstr/B | 95%CI            |
|---------|--------------|---------------|----------------------|----------|------------------|
| silesia | 5.868        | 6.087         | **+0.216 (+3.68%)**  | +0.599   | [+0.200,+0.225]  |
| monorepo| 5.052        | 5.209         | **+0.155 (+3.06%)**  | +0.291   | [+0.138,+0.165]  |
| nasa    | 3.336        | 3.423         | **+0.087 (+2.62%)**  | +0.224   | [+0.079,+0.094]  |
⇒ Packing buys only 0.087–0.216 cyc/B (SMALL). B abandons this. Residual gap to igzip
is +1.50/+2.07/+1.75 — so the packing loss is ~1/7th the gap on silesia.
RE-VERIFY: `BIN_A=/root/bin/gzippy-new-native BIN_B=/root/bin/gzippy-packOFF PIN=4 REPS=21 CORPORA="silesia monorepo nasa" SKIP_STRESS=1 bash /root/distpreload-harness/_distpreload_paired_guest.sh`

### MEASUREMENT 3 — B-SPIKE: measured bound on B's achievable cyc/B
SPIKE = throwaway partial-B (gz-flagbit-new patched: lut SINGLE_SYM + asm cnt-extract
hardcoded `mov t3,1` since cnt≡1 in SINGLE mode → removes the mov/shr/and unpack
cnt-extract chain that B eliminates). BYTE-EXACT silesia/nasa/monorepo; distinct
binary (sha 9be95b74≠packON 891c9925) — non-inert. N=21 paired vs packON(=current
flag-bit kernel), unstressed. medΔ=(spike−packON), all SIGNIF, self-tests PASS:
| corpus  | packON cyc/B | spike cyc/B | spike−packON medΔ | 95%CI            | vs igzip (banked) |
|---------|--------------|-------------|-------------------|------------------|-------------------|
| silesia | 5.869        | 5.954       | **+0.091 SLOWER** | [+0.078,+0.096]  | +1.59 (+36%) still|
| monorepo| 5.089        | 5.166       | **+0.078 SLOWER** | [+0.060,+0.095]  | +2.21 (+75%) still|
| nasa    | 3.330        | 3.340       | **+0.015 SLOWER** | [+0.005,+0.021]  | +1.76 (+112%) still|
⇒ The partial-B spike is SIGNIF SLOWER than the current packed kernel on ALL 3 corpora.

DECOMPOSITION (derived; packON A1 medians stable 5.868↔5.869 across the two paired runs):
  spike−packOFF = unpack(cnt-extract) RECOVERY = {sil −0.125, mono −0.077, nasa −0.072}.
  So removing the cnt-extract recovered 0.072–0.125 cyc/B — but going single-mode COST
  0.087–0.216 (M2). NET partial-B = packing_loss − unpack_recovery = +0.015..+0.091 WORSE.

⇒ **MEASURED net-B ≈ NEGATIVE-to-BREAKEVEN, NOT the 1.34 ceiling.** Full B (fat direct
  table) removes ~2 MORE unpack ops than the spike (byte-mask + bc-extract), so could
  recover a bit more — but to even BREAK EVEN with the current packed kernel on silesia
  it must recover another +0.091 from ~2 ops, comparable to what the 3-op cnt-extract
  just bought. Best realistic case: full B ≈ break-even-to-small-win vs CURRENT — it
  does NOT approach igzip (spike vs igzip is UNCHANGED at +36/75/112%).

WHY the 1.34 ceiling is a MIRAGE for B-as-scoped: igzip is ALSO one-symbol-per-iter yet
hits 3.98 cyc/B; gzippy single-mode (packOFF) is 6.09. The 2.1 cyc/B gap between two
one-symbol-per-iter loops is NOT the table format — it is igzip's whole leaner
per-iteration machinery (bit-reader cadence, refill, copy, codegen). B (table-format
swap) does not touch that, so it cannot reach 3.98. The 1.34 removal-oracle bounded a
DIFFERENT thing (igzip's entire kernel), not the unpack-removal lever.

---

Mission: make gzippy-native (pure-Rust, FFI-off) **T1 single-member gzip DECODE**
measurably FASTER than igzip (ISA-L), byte-exact, gated. Single-arch Intel = NOT LAW
(AMD/Zen2 replication owed). T1 single-core only — no T4/T8 extrapolation.

Branch: `perf/igzip-full-rewrite`. Mac worktree (edit/commit/push only — aarch64,
CANNOT run the asm): `/home/user/www/gzippy/.claude/worktrees/agent-a8069a92d914fcef3`.
Guest (ONLY x86_64/BMI2 measure box): `ssh -J REDACTED_IP root@REDACTED_IP`.
Guest worktrees: gzippy(B)=/root/gz-fullrewrite (kernel 2b10aa48 dist-preload),
baseline=/root/gz-baseline (8383a2eb). igzip=/usr/bin/igzip (ISA-L 2.31.1).
Harness on guest: /root/distpreload-harness/.

## COMMITS THIS MISSION
- 2c135d07 — deliverable #0: commit orphaned paired harness (analyzer+memstress+driver) to scripts/bench.
- 2e01dd4f — deliverable #1: add igzip arm `scripts/bench/_gzippy_vs_igzip_paired_guest.sh`.
- (HEAD)   — Step-B technique #1: fuse `lea-1`+`shl3` shift into one `lea [t3*8-8]`.
            BYTE-EXACT (sha 3 corpora×T1/T4/T8, proptest 60k, c2/c3 asm-vs-ref diffs).
            Δinstr/byte=-0.186 (silesia, deterministic + paired); cyc/byte TIE
            (silesia medΔ=-0.040, CI=[-0.0475,-0.0216] excl 0 but p=0.018 fails p<0.01;
            nasa wash). KEPT on byte-exact license; gap to igzip NOT closed (still +39.5%).

## STEP-B TECHNIQUE LOG (gated, Intel trainer cpu4, N=21 paired, /dev/null sink)
| # | technique | byte-exact | Δinstr/B (sil) | cyc/B medΔ (sil) unstressed | p | verdict | kept? |
|---|-----------|-----------|----------------|------------------|---|---------|-------|
| 1 | fuse shift `lea[t3*8-8]` (was lea-1+shl3) | PASS | -0.186 | -0.040 [CI-0] | 0.018 | TIE (fails p<0.01) | YES (byte-exact) |
| 2 | FLAG-BIT discriminator (`test entry,0x1000000;jnz` off build-time trailing-class bit 24; removes shift-lea+and+shrx+cmp from literal discriminator chain) | PASS | -0.279 | **-0.099 (-1.67%)** [CI -0.103,-0.094] | 6.4e-05 | **SIGNIF-faster UNSTRESSED** (sil+nasa+mono p<0.01, CI excl 0); WASH/TIE under bandwidth stressor (LLC 91%) | YES |
  Re-verify #1/#2: BIN_A=/root/bin/gzippy-base-native BIN_B=/root/bin/gzippy-new-native
  PIN=4 REPS=21 CORPORA="silesia nasa monorepo" SKIP_STRESS=1
  GZIPPY_FORCE_PARALLEL_SM=1 bash /root/distpreload-harness/_distpreload_paired_guest.sh
  (stressor arm: SKIP_STRESS=0 CORPORA="silesia nasa")

  TECHNIQUE #2 RESULT (2026-06-18, this session):
    - byte-exact: native sha grid silesia/nasa/monorepo/squishy × T1/T4/T8 == gzip;
      c1/c2/c3 asm-vs-ref differential PASS (x86_64); proptest 60k prop_structured_roundtrip
      PASS; new lut_huffman flag-invariant test PASS (flag bit24 == trailing>=256 for
      every short entry). exit profile IDENTICAL to base (entries=24128, reclass_dist=21330,
      reclass_eob=2796) — same paths, fewer instrs.
    - UNSTRESSED gated cyc/byte (base-vs-new, self-test PASS all 3, mono self-test flagged
      a -0.0044 rig bias dwarfed by the -0.062 effect):
        silesia  medΔ=-0.0993 (-1.67%)  CI[-0.103,-0.094]  p=6.4e-05  SIGNIF-faster
        nasa     medΔ=-0.0112 (-0.33%)  CI[-0.022,-0.005]  p=0.0067   SIGNIF-faster
        monorepo medΔ=-0.0620 (-1.20%)  CI[-0.079,-0.043]  p=0.00085  SIGNIF-faster
        ΔIPC sil -0.010, Δinstr/B sil -0.279 (the removed shift+extract+cmp).
    - STRESSED (memstress 14T, LLC 91-92%): WASH/TIE
        silesia medΔ=-0.013 CI[-0.245,+0.093] p=0.465; nasa medΔ=+0.014 p=0.627.
        Δinstr/B STILL -0.28 (sil) stressed — the WORK is removed in both regimes; the
        cyc/byte win only surfaces when NOT bandwidth-saturated (compute-bound regime).
    - VERDICT: real CRITICAL-CHAIN win in the compute-bound regime (contrast #1's
      load-shadowed TIE — this time cyc/byte moved with instr/byte because the removed
      ops fed the discriminator branch the Gate-2 perturbation proved is on the critical
      recurrence). Does NOT meet the strict "stressor-stable" arm. KEPT on byte-exact +
      compute-bound win. Single-arch Intel = NOT-YET-LAW (AMD owed).
  LESSON #1 (still valid): removing a PREDICTABLE, load-shadowed instr drops instr/byte
  but NOT cyc/byte — the shl was overlapped behind the table load. Technique #2 confirms
  the inverse: cutting the CRITICAL-PATH discriminator chain DOES move cyc/byte (unstressed).

## GATE-2 CAUSAL PERTURBATION — IS THE UNPACK/CLASSIFY CHAIN ON THE CRITICAL RECURRENCE? (2026-06-18)
DECISIVE TEST that gates whether rewrite B (igzip-style fat DIRECT table, ~0 unpack)
is worth funding. Method (STRONG tier — Gate-2 causal perturbation): in a THROWAWAY
worktree (/root/gz-perturb, base 2b10aa48 — identical shrx→cmp critical segment;
REMOVED after run), inserted K serially-DEPENDENT, value-NEUTRAL ALU ops
(`rol $1,%r10`/`ror $1,%r10` pairs, all on t5) BETWEEN the trailing-symbol extraction
`shrx {t5},{t5},{t4}` and the literal/non-literal discriminator `cmp {t5:e},255`,
extending the loop-carried recurrence that feeds the discriminator branch.
CONTROLS PASSED: all 4 builds byte-exact (sha==zcat silesia ref) + kernel-engaged
(entries=24128 identical) + distinct binaries; disasm-verified the chain is a genuine
DEPENDENT serial chain on %r10 sitting exactly between the shrx and `cmp $0xff,%r10d`
+ `ja` (rol/ror count scales 8→10→12→16 = +K); paired self-test (A2-A1) PASS.
Harness: `_distpreload_paired_guest.sh`, PIN=4 REPS=21 SKIP_STRESS=1, /dev/null sink.

SLOPE TABLE (medΔ cyc/byte vs K=0, all p=6.412e-05, CI excludes 0):
| K (dep ops) | silesia medΔ (per-op)   | nasa medΔ (per-op)     |
|-------------|-------------------------|------------------------|
| 2           | +0.280 (+4.6%) [0.140]  | +0.070 (+2.1%) [0.035] |
| 4           | +0.686 (+11.3%) [0.171] | +0.204 (+6.1%) [0.051] |
| 8           | +1.380 (+22.8%) [0.172] | +0.375 (+11.2%) [0.047]|

VERDICT: **PROPORTIONAL slope on BOTH corpora** (monotone, ~linear; silesia
doubling-K → ~2.0× Δ, ~0.17 cyc/B per op; nasa ~0.05 cyc/B per op). ⇒ the
shrx-fed t5-extraction → discriminator chain (the gz-specific UNPACK/classify the
STATE'd ~21% region) IS on the critical recurrence — NOT load-shadowed (contrast
technique #1's load-shadowed shl, which TIE'd). Eliminating that extraction
(rewrite B: direct table yields byte+len, ~0 unpack) therefore has REAL cyc/byte
headroom. The unpack cost is concentrated on entropy-DENSE/literal-heavy data
(silesia 0.17/op) vs backref-heavy/low-entropy (nasa 0.05/op) — i.e. B's payoff is
LARGEST exactly where gzippy's gap is largest (silesia +39.5%).

CAVEAT (Gate-2 rule: slow-down slope ≠ speed-up ceiling): this proves the chain is
ON the critical path (necessary), NOT how much B's shorter chain recovers. The
removal-ORACLE that BOUNDS the win already exists and is banked: igzip's
decode_huffman kernel = 3.98 cyc/B vs gzippy run_contig 5.32 cyc/B (silesia, Step A
iii) ⇒ up to ~1.34 cyc/B recoverable, and igzip pays ~0 unpack. So: perturbation
(chain on critical path) + igzip removal-oracle (1.34 cyc/B ceiling) ⇒ B is
JUSTIFIED. Single-arch Intel = NOT-YET-LAW (AMD/Zen2 owed). monorepo 3rd-corpus owed
(optional; 2-corpus spread already concordant).

RECOMMENDATION FOR USER R3 (fund rewrite B?): **YES, justified** — gated proportional
slope (both corpora) + a measured igzip removal-oracle ceiling (~1.34 cyc/B). BUT do
the CHEAPER step first (pre-registered NEXT #1: flag-bit discriminator — same critical
chain, ~1/10th the work, no table re-layout): if a single `test entry,FLAG; jnz` off
the LOAD (no shrx-extract) recovers a significant fraction of the slope, B's full
table-format rewrite may be unnecessary. Sequence: flag-bit (NEXT #1) → re-measure →
THEN decide B with the residual.

## THE INSTRUMENT (deliverable #1) — Gate-0 SELF-VALIDATED, PASS
`scripts/bench/_gzippy_vs_igzip_paired_guest.sh` (reuses committed `_distpreload_paired_analyze.py`).
arm A(A1,A2)=igzip, arm B=gzippy-native (ParallelSM @ T1). medΔ=(B-A1) cyc/byte;
NEGATIVE => gzippy faster. Gate-0 verified live: gzippy run_contig KERN entries>0
(monorepo 8299, nasa 25399), igzip --version printed, BOTH arms sha==zcat ref==each
other, same /dev/null sink + same pin(cpu4), GHz spread <0.07%, A2-A1 self-test ~0.
Run: `PIN=4 REPS=21 CORPORA="silesia monorepo nasa" /root/distpreload-harness/_gzippy_vs_igzip_paired_guest.sh`

## THE GAP TO CLOSE — gzippy-native T1 vs igzip cyc/byte (whole-process, /dev/null sink) — GATED
FULL N=21 paired, Wilcoxon p, bootstrap CI, A2-A1 self-test, + bandwidth stressor.
Intel i7-13700T LXC (cpu4 pinned). All cells p=6.4e-5, CI excludes 0, self-test PASS
(except stressed-nasa, noted). medΔ=(B-A1)>0 means gzippy SLOWER. THE GAP IS LARGE:

UNSTRESSED (clean) — THE number to close:
START (base af53f95e, banked):
| corpus   | igzip cyc/B | gzippy cyc/B | medΔ (B-A1)      | Δinstr/byte | ΔIPC   |
|----------|-------------|--------------|------------------|-------------|--------|
| silesia  | 4.30        | 6.00         | +1.70 (+39.5%)   | +~2.3       | -0.27  |
| monorepo | 2.97        | 5.17         | +2.20 (+73.9%)   | +2.83       | -0.43  |
| nasa     | 1.58        | 3.43         | +1.84 (+116.5%)  | +2.08       | -0.64  |

CURRENT after technique #2 FLAG-BIT (new-native vs igzip, 2026-06-18, unstressed,
all self-tests PASS, N=21 paired):
| corpus   | igzip cyc/B | gzippy cyc/B | medΔ (B-A1)      | Δinstr/byte | ΔIPC   |
|----------|-------------|--------------|------------------|-------------|--------|
| silesia  | 4.367       | 5.871        | +1.499 (+34.3%)  | +1.86       | -0.35  |
| nasa     | 1.576       | 3.331        | +1.754 (+111.3%) | +1.98       | -0.64  |
| monorepo | 2.956       | 5.033        | +2.070 (+70.0%)  | +2.73       | -0.41  |
  ⇒ gap CLOSED by the flag-bit: silesia +39.5%→+34.3% (~0.10 cyc/B), monorepo
    +73.9%→+70.0% (~0.06), nasa ~+0.01. RESIDUAL gap = +1.50/+1.75/+2.07 cyc/B.
    Removal-oracle ceiling was ~1.34 cyc/B (silesia kernel 5.32→3.98); flag-bit
    recovered ~0.10 of it ⇒ ~7% of the headroom captured, ~1.24 cyc/B STILL on the
    table ⇒ rewrite B (fat DIRECT table, ~0 unpack) remains justified by the residual.

STRESSED (memstress 14T, LLC-miss ~42-45%): gap HELD/GREW (silesia +39.5%, monorepo
+108%, nasa +179%) ⇒ NOT a bandwidth artifact; it is real work-volume. (stressed-nasa
self-test FAILED 2.9% — that ONE cell's precision is degraded; effect dwarfs it; use
the unstressed nasa cell as the clean number.)

GATED FACTS (this commit, Intel-only, NOT-YET-LAW — AMD owed):
1. gzippy-native T1 is **40-116% SLOWER than igzip** (cyc/byte), gated, 3 corpora.
2. The gap is **INSTRUCTION-DOMINATED**: gzippy retires **+2.0 to +2.8 more instr/byte**
   than igzip (ΔIPC also negative). This is WORK VOLUME, not micro-latency — a preload
   tweak will NOT close it; the lever is REDUCING instr/byte in the hot loop.
3. **run_contig = 85.36% of T1 self-time** (perf, symboled binary, silesia). Scaffold
   (block finder / marker machinery / apply_window / CRC) is only ~15%. So the excess
   instructions ARE in run_contig — the mission's kernel focus is correctly aimed.
4. Hot per-symbol economy in run_contig (perf annotate, top self-time instrs):
   multiple classify branches PER SYMBOL — `cmp $0x100`(EOB) 5.8%, `cmp $0xff`(literal-
   range) 3.6%, `test $0x2000000`(marker/reclass discriminator) 2.8%, plus two table
   loads `mov (%r11,%rXX,4)` (6.3%+) and the engaged MOVDQU backref copy (4.5%x2,
   8383a2eb is LIVE). igzip's loop_block collapses this to ONE speculative store + ONE
   discriminator branch — THAT delta is the instruction excess to attack.

## STEP A — AIM CONFIRMED (2026-06-18, guest trainer cpu4, symboled bin rebuilt today)
Localization re-run fresh; all three sub-checks PASS and converge on "the excess is in
the hot run_contig classify+decode loop, addressable."

(i) RECLASS RATIO — cold exits are NOT the cost. GZIPPY_VERBOSE=1 GZIPPY_ASM_STATS=1:
| corpus   | entries | exit_reclass(tag0) | reclass_dist | reclass_eob | asm_bytes/entry |
|----------|---------|--------------------|--------------|-------------|-----------------|
| silesia  | 24128   | 0                  | 21330        | 2796        | ~8770           |
| monorepo | 8299    | 0                  | 8130         | 166         | ~6118           |
| nasa     | 25399   | 0                  | 25037        | 351         | ~8077           |
  Generic invalid/oversize exit (tag 0) = 0 everywhere. The dominant exit is
  reclass_dist (subtable/raw0 dist → Rust), BUT asm_bytes/entry is ~6-9 KB: the kernel
  decodes thousands of CLEAN bytes between each bail, so per-byte cold-exit overhead is
  negligible. The excess instr/byte is in the hot loop. => loop_block port is aimed right.

(ii) PER-REGION (perf annotate -F4000, silesia, hottest run_contig instrs, self%):
  - Packed-symbol UNPACK ALU chain (gz's multi-sym short-entry format tax, no igzip
    counterpart): test 0x2000000 3.27, mov t2,t1 4.31, shl $3 (shift=8*(cnt-1)) 3.81,
    and 0x1FFFFFF 2.96, mov 2.23, lea-1 1.38, shrx 1.12, cmp 0xff 1.92  => ~21%
  - Two table-load preloads: mov(%r11,%r14,4) 5.63 + mov(%r11,%r12,4) 5.55 => ~11%
    (IRREDUCIBLE — igzip's decode_next_sym has the identical 2 loads)
  - MOVDQU backref copy: load 4.69 + store 4.10 + jle 2.03 => ~11%
  - Non-literal arm: cmp 0x100 (EOB) 5.12, cmp 0x200 1.28, dist preload/decode/copy body

(iii) SYMBOL-RESTRICTED KERNEL-vs-KERNEL cyc/byte (perf stat cycles × self%, silesia):
  - gzippy run_contig self=86.64%, total=1,302,094,139 cyc / 211,968,000 B => 5.32 cyc/B
  - igzip decode_huffman_code_block_stateless_04 self=90.31%, total=934,030,780 cyc
    => 3.98 cyc/B
  - KERNEL GAP = +1.34 cyc/B (+33.7%) — tracks the whole-process silesia +39.5%, so the
    excess really is IN the kernel. Whole-proc instr/byte: gzippy 13.73 vs igzip 11.38
    (+2.35), consistent with banked +~2.3.

  VERDICT (Step A): the +1.34 cyc/B kernel deficit decomposes as ~21% packed-symbol
  UNPACK ALU + ~11% backref copy + ~11% irreducible table loads. The single addressable
  lever with NO igzip counterpart is the packed-multi-symbol UNPACK chain (cnt extract,
  shift=8*(cnt-1), shrx, masks). igzip's table format yields the decoded byte(s)+length
  DIRECTLY, paying ~0 unpack ALU — that ~21% is the instruction excess. NOTE: attacking
  it = changing the short-entry TABLE FORMAT (build side in lut_huffman) in lockstep, a
  STRUCTURAL change, not a peephole; high byte-exact risk, must be one gated commit.

## RESUME POINT (2026-06-18 PM, end of B-SIZING session) — READ FIRST
DONE this session: the 3 B-sizing GATING measurements (M1/M2/M3 above) that price
rewrite B for the USER R3 decision. Source on guest restored to mission tip; spike
removed; disk restored (1.1G); no stressors/pinning leftover; governor untouched
(powersave); NO main push. Kept binaries: /root/bin/gzippy-new-native (=mission tip
packON, sha 891c9925), /root/bin/gzippy-base-native (af53f95e), /root/bin/gzippy-packOFF
(M2 SINGLE-mode toggle, sha ff477774). NOT committed: nothing new to source — only this
STATE file edited.

### RECOMMENDATION FOR USER R3 (fund the multi-hour rewrite B?): **NO — do NOT fund B as scoped.**
GATED-HYPOTHESIS (Intel-only, NOT-YET-LAW; AMD/Zen2 owed). The prior STATE rec ("B
remains justified by the residual / 1.34 ceiling") is OVERTURNED by M3:
  - M1: the real (unstressed) regime is COMPUTE-bound (silesia DRAM-stall 0.69%, TMA
    mem-bound 7%), so headroom is real — but that only says SOME compute lever exists.
  - M2: multi-symbol packing currently BUYS 0.087–0.216 cyc/B; B abandons it.
  - M3: a byte-exact partial-B spike (one-sym-per-iter + unpack cnt-extract removed) is
    SIGNIF SLOWER than the current packed kernel on ALL 3 corpora (+0.015..+0.091 cyc/B).
    The unpack removal recovered only 0.072–0.125 — LESS than the packing it costs.
  - ⇒ MEASURED net-B is negative-to-breakeven vs current, NOT the 1.34 ceiling. The
    table-format swap cannot close the igzip gap: igzip is ALSO one-sym-per-iter at 3.98
    while gzippy single-mode is 6.09 — the 2.1 cyc/B is igzip's whole leaner per-iteration
    loop, which B does not touch.
  - The igzip residual (+1.50/+2.07/+1.75 cyc/B) is therefore in the PER-ITERATION
    MACHINERY (bit-reader/refill cadence, copy, codegen), not the short-entry table format.

### NEXT (if the campaign continues toward the igzip gap):
  - DO NOT start B (fat-direct-table). It is measured net-negative-to-breakeven.
  - The real lever is converging gzippy's per-iteration loop on igzip's (single
    speculative store + one discriminator + igzip's refill cadence) — i.e. the gap
    between two one-symbol-per-iter loops (6.09 vs 3.98). Size THAT before committing:
    a removal/perturbation on the refill+copy+back-edge overhead, not the table format.
  - AMD/Zen2 replication of M1/M2/M3 is owed before any of this is LAW.
  - RE-VERIFY M3: rebuild spike (patch lut:1041 SINGLE + asm 426-428 `mov t3,1`; see
    /tmp gone — re-derive from this STATE), or simpler re-confirm via packOFF: packOFF
    is already SIGNIF slower than packON (M2), and the spike only recovered a fraction.

## (prior) RESUME POINT (2026-06-18, end of technique #2 = NEXT #1 flag-bit)
DONE this session: NEXT #1 (flag-bit discriminator) IMPLEMENTED + GATED + KEPT.
  - byte-exact PROVEN on x86_64 guest (sha grid native, c1/c2/c3 asm-vs-ref, proptest
    60k, builder flag-invariant test). KEPT.
  - cyc/byte: SIGNIF-faster UNSTRESSED (sil -1.67% p=6.4e-5, nasa -0.33% p=0.0067,
    mono -1.20% p=8.5e-4, self-tests PASS except mono small rig-bias); WASH/TIE under
    bandwidth stressor (LLC 91%). Does NOT meet strict "stressor-stable" arm; KEPT on
    byte-exact + compute-bound critical-chain win.
  - NEW residual gap vs igzip: silesia +1.50 (+34.3%), nasa +1.75 (+111%), monorepo
    +2.07 (+70%) cyc/B. Flag-bit captured ~7% of the silesia removal-oracle headroom.
COMMIT: technique #2 source (lut_huffman flag bit + builder + invariant test;
  asm_kernel flag-bit discriminator + 49: trailing-recovery shim; lut_bulk
  symbol→code length fix) pushed to perf/igzip-full-rewrite.
GUEST BINARIES (kept for next A/B): /root/bin/gzippy-base-native (af53f95e),
  /root/bin/gzippy-new-native (af53f95e+flagbit). Worktrees: /root/gz-flagbit-new
  (patched src), /root/gz-flagbit-base (clean af53f95e). Re-verify cmds in the
  technique log above.

RECOMMENDATION (B decision — for USER R3): the flag-bit did NOT capture most of the
headroom (~0.10 of ~1.34 cyc/B on silesia). Residual gap to igzip is STILL +1.50 to
+2.07 cyc/B (+34-70%). ⇒ rewrite B (fat DIRECT one-symbol-per-iter table, ~0 unpack
ALU) REMAINS justified by the residual. CAVEAT: the stressor wash shows much of the
kernel time is memory-bound under contention; B's instruction reduction (like #2's)
will similarly wash when bandwidth-saturated — B's payoff is bounded to the
compute-bound regime (which is where the removal-oracle 1.34 cyc/B ceiling was
measured). NEXT step before B: optionally extend the flag idea — also bake a
"length-vs-EOB-vs-reclass" sub-class into spare entry bits to shrink the 50: arm —
but the bigger lever is B. DO NOT begin B without USER R3.

## NEXT (planned, in priority order) — for the iteration phase (deliverable #2)
REVISED after technique #1's TIE: the prize is CRITICAL-PATH cyc/byte, NOT retired
instruction count. Technique #1 cut -0.186 instr/byte yet TIE'd cyc/byte because the
removed shl was load-shadowed (overlapped behind the table-load latency). Peephole
instruction-shaving of already-overlapped ops will keep TIE-ing. The remaining cyc/byte
deficit (+1.34 cyc/B kernel) lives in (a) the data-dependent classify dep-chain that
FEEDS the load and the cmp (the `shrx`-fed trailing-symbol extraction → `cmp 255`
branch; negative ΔIPC says this mispredicts/serializes), and (b) the ~21% packed-symbol
UNPACK ALU that has no igzip counterpart. So:
1. (HIGHEST VoI) Attack the CRITICAL PATH, not the instruction count. The dep chain is
   short_tbl load -> (bc/cnt/shift extraction) -> shrx -> cmp 255 -> branch. Shorten the
   chain LENGTH or break the misprediction: e.g. derive the literal-vs-nonliteral
   discriminator from a flag BIT already in the loaded entry (a single `test`, no
   shrx-extract), so the branch resolves off the load directly instead of waiting on the
   extraction chain. Requires a table-format bit (build side lut_huffman) — STRUCTURAL,
   one gated commit, ref in lockstep. Pre-register success = cyc/byte medΔ<0 p<0.01.
2. The packed-multi-symbol short-entry format itself: evaluate whether multi-symbol
   packing (decode 2-3 lits/iter, pay UNPACK ALU) actually beats igzip's
   one-symbol-per-iter + fatter DIRECT table (decoded byte+len in the entry, ~0 unpack).
   This is the core convergence question — likely the real lever, but a large rewrite +
   STRATEGIC fork: escalate to supervisor/user before committing the rewrite (R3).
3. Update `run_contig_ref` / `run_contig_ref_biased` in LOCKSTEP with every asm change
   (X1-X5 exit-state + IN_MARGIN bit-exact-refill contract).
GATE (each technique, ONE per commit): byte-exact (sha 3 corpora×T1/T4/T8 + proptest≥60k
+ c2/c3 asm-vs-ref diffs) THEN paired N≥21 cyc/byte vs prior AND vs igzip. Build native
flavor: `RUSTFLAGS="-C target-cpu=native" cargo build --release --no-default-features
--features pure-rust-inflate`. NOTE: local Mac pre-commit hook is BROKEN (uninitialized
vendor submodules) — commit on the guest, or `git commit --no-verify` for non-perf
metadata; CI still runs the real checks.

## TOOLING STATE (for next turn)
- Instrument: `/root/distpreload-harness/_gzippy_vs_igzip_paired_guest.sh` (committed
  in scripts/bench). Re-run gap: `PIN=4 REPS=21 CORPORA="silesia monorepo nasa" /root/distpreload-harness/_gzippy_vs_igzip_paired_guest.sh`
- Symboled native binary for perf/annotate: `/tmp/symtarget/release/gzippy` on guest
  (byte-correct, built from gz-fullrewrite @ 2b10aa48; rebuild after any kernel edit).
  Profile: `perf record -F6000 ... ; perf annotate -s ...run_contig`.
  NOTE: build needs strip=false debug=2 WITHOUT force-frame-pointers (FP collides with
  the register-hungry asm kernel: "inline assembly requires more registers").

## DONE-CRITERION (do NOT self-bless; report as gated-HYPOTHESIS + re-verify cmd)
gzippy T1 cyc/byte ≤ igzip with paired p<0.01 + bootstrap CI excluding 0 + margin
surviving the bandwidth stressor, on silesia AND ≥1 more corpus; + wall-time
confirmation if a quiet window appears. AMD/Zen2 replication owed for LAW.
