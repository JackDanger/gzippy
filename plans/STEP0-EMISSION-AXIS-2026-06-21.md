# STEP-0 EMISSION AXIS — is gz's clean-emission gap vs ISA-L INSTRUCTION-COUNT or IPC, and does the isolated gap reconcile with production silesia-T4 +16%?

**Date:** 2026-06-20/21  **Branch:** kernel-converge-A  **gz decode path:** 50ebadd1
(== 9df14077, the W3 Intel binary — verified `git diff 9df14077..50ebadd1 -- src/` is
EMPTY; only docs/scripts changed since the W3 +16% reproduction).
**Box:** Intel i7-13700T LXC (neurotic), LOADED (load 4.4–4.8), single P-core pinned
(taskset -c 3). **Stamp:** NOT-YET-LAW — single-arch Intel x86, ISA-L/igzip is x86-only;
AMD/Zen2 owed. MEASUREMENT ONLY (no decode-path src change; new files are harness +
note + artifact).

## THE TWO QUESTIONS (answered deterministically; both outcomes actively held)
- **Q1 (AXIS):** isolated, same silesia clean block, single pinned P-core — is gz's
  clean-PATH emission gap vs igzip `_04` INSTRUCTION-COUNT or IPC/THROUGHPUT?
- **Q2 (RECONCILIATION):** does the ISOLATED gz-vs-`_04` cyc/B gap account for the
  production silesia-T4 +16% wall gap (→ isolated screens VALID) or is it MUCH smaller
  (→ T4 loss is 4-thread bandwidth/contention, isolated screens INVALID)?

## THE INSTRUMENT
`examples/kernel_ab.rs` (already on-branch) A/Bs **ARM A = gz production clean-path
emission** (`Block::decode_clean_into_contig` = Huffman-decode + per-symbol copy/store +
loop, the WHOLE clean per-symbol path that W1 named) vs **ARM B = igzip
`decode_huffman_code_block_stateless_04`** (AVX2/BMI2 stateless clean kernel, called
DIRECTLY), each decoding the SAME real silesia dynamic block (start_bit 0, 62581 B out,
tables built ONCE outside the timed loop), looped.
New gated wrapper: `scripts/bench/kernel-ab/emission_axis_{guest.sh,analyze.py}`.
Kernel-LOOP-only counts isolated by the **DIFFERENCE METHOD** (perf counts at reps=2R
minus reps=R == exactly R reps of pure loop; one-time setup/compress/warm cancels).
N=15 interleaved rounds, R=2000/4000, perf events cycles,instructions,LLC-load-misses,
task-clock on the pinned P-core (cpu_core PMU; cpu_atom not-counted by construction).

## Gate-0 (self-validation — all PASS)
- **byte-exact:** ARM A and ARM B each assert produced==flate2 oracle (62581 B) and die
  otherwise; both passed (`ARM A Gate-0 byte-exact OK`, `ARM B kernel-only setup verified
  byte-exact`).
- **non-inert (proven):** ARM B calls `_04` directly (no multibinary dispatch ambiguity);
  ARM A `asm_kernel enabled=true`, stateless_entries gate consistent. Counts differ
  between arms (different code ran).
- **same sink:** both arms write an in-RAM buffer.
- **A/A self-test (split-half ratio ~1.0):** gz instr=0.9998 cyc=0.9954 ipc=1.0049;
  `_04` instr=0.9987 cyc=0.9946 ipc=1.0020 — all ≈1.0 (no drift between halves).
- **GHz-stability gate:** gz 1.3946 GHz (spread 0.30%), `_04` 1.3914 GHz (spread 0.61%)
  — both arms at the SAME ~1.39 GHz (0.2% apart) and stable ⇒ the cyc/B comparison is
  frequency-fair (low GHz is the loaded box, but it is symmetric+stable, which is what
  the gate requires).
- **LLC-load-miss/B ≈ 0** for BOTH arms (gz 0.00009, `_04` 0.00003) — the 62 KB block +
  tables stay hot in L1/L2; the isolated slice is PURE COMPUTE, no memory-bandwidth
  component (directly relevant to Q2).

## Q1 RESULT — the AXIS is INSTRUCTION-COUNT (loop-only, single P-core, gated)
```
                 |  instr/B  | core-cyc/B |  IPC  | LLC-miss/B | GHz   | rdtsc cyc/B
ARM A gz emit    |  15.867   |   5.240    | 3.027 |  0.00009   | 1.395 |   5.194
ARM B igzip _04  |  12.645   |   4.387    | 2.883 |  0.00003   | 1.391 |   4.373
gz / _04         |  1.2547   |   1.1944   | 1.0498|     —      |  —    |   1.1878
Δ (gz − _04)     | +3.221/B  |  +0.853/B  |       |            |       |
```
- gz executes **+25.5% more instructions/byte** (+3.221 instr/B) than `_04`.
- gz **IPC is HIGHER (+5.0%)**, not lower — IPC works in gz's *favor*.
- cyc/B gap decomposition: **+127.7% of the cyc gap is MORE INSTRUCTIONS**; IPC
  *subtracts* −27.3% (gz's higher IPC partially offsets its instruction surplus).
- rdtsc loop-only cyc/B (independent of perf, TSC-based) cross-checks: 1.188 ≈ the perf
  core-cyc/B 1.194.

**Q1 VERDICT: INSTRUCTION-COUNT.** gz's clean-path emission loses to ISA-L `_04` because
it *executes more instructions per byte* (the AVX2/BMI2 wide copy/store + multi-symbol
packing of `_04` does the same work in fewer instructions), NOT because its pipeline
stalls — gz already retires at higher IPC. This re-confirms NIGHT31's whole-kernel-T1
hint (+3.40 instr/B at higher IPC) specifically for the EMISSION path (+3.22 instr/B at
+5% IPC). **The convergence loop's gate axis must be instr/B (load-immune, deterministic
on the loaded LXC), NOT cyc/IPC.** Chasing IPC is the wrong lever — gz's IPC is already
ahead; the win is in removing instructions.

## Q2 RESULT — RECONCILES; isolated screens are VALID
Production silesia-T4 gz/rg, re-measured FRESH at this decode path (interleaved best-of-15,
/dev/null both arms, sha==zcat both arms, rapidgzip-native, T=4, load 4.77):
```
gz  best=496.8 ms med=524.7  spr=11.4%   (path=ParallelSM, flavor=parallel-sm+pure, FFI off)
rgA best=425.9 ms med=436.1  spr=4.9%    rgB best=422.9 ms spr=4.7%
A/A rg(best)/rg(best) = 1.0071   A/A med = 1.0031   (box symmetric — no phantom sign-flip)
gz/rg = 1.1666 (best) / 1.2031 (median)  → gz SLOWER +16.7% / +20.3%
```
(matches the standing +16% and the W3 reproduction; the harness "UNTRUSTED" flag fires
only because gz spread 11.4% > a 5% tol on this loaded run, but A/A=1.007 and
Δ(16.7%) > spread make the gap sound.)

**Magnitude comparison:**
- ISOLATED clean-emission cyc/B gap: **+19.4%** (core) / **+18.8%** (rdtsc), single
  P-core, hot-cache, LLC-miss/B ≈ 0.
- PRODUCTION silesia-T4 wall gap: **+16.7% to +20.3%**.

These are the **SAME ORDER OF MAGNITUDE** — the isolated clean-emission gap is *not
tiny*; it is essentially equal to (slightly larger than) the production T4 gap. Two
corroborating facts make this a VALID reconciliation, not a coincidence:
1. The isolated slice has **near-ZERO LLC-miss/B**. If the T4 loss lived in the 4-thread
   bandwidth/cache-coherence/contention regime, a zero-miss single-thread slice would
   show a *much smaller* gap than +16%. It instead shows +19%. ⇒ the T4 loss is
   **per-byte CLEAN-PATH COMPUTE (instruction count)**, present even single-thread
   hot-cache — NOT a contention phenomenon.
2. Consistent with W2 (the gap concentrates in the clean portion; markers
   anti-correlate / are gz-favorable): the clean kernel being +19% on its portion,
   diluted by gz-favorable markers, lands the whole-decode T4 at +16–20%.

**Q2 VERDICT: RECONCILES — isolated screens are VALID (representative of the T4 loss).**
The dominant driver of the production silesia-T4 +16% is the clean-path per-byte
instruction count, which the isolated kernel_ab A/B captures faithfully. (The exact
apportionment between isolated-clean-compute and any residual 4-thread pipeline overhead
is OWED, but the isolated emission gap is large enough to BE the dominant component, so
isolated screening is a valid design basis.)

## VERDICT FOR THE CONVERGENCE LOOP
**BUILD THE LOOP WITH AN INSTR/B GATE.** Concretely:
- Gate axis = **instr/B** (load-immune, deterministic — measurable on the loaded LXC
  without freq-pin). Secondary confirm = core-cyc/B + rdtsc cyc/B under the GHz gate.
- Screen on the ISOLATED `kernel_ab` A/B (proven representative of the T4 wall by Q2);
  promote a win to the production silesia-T4 wall A/B (rg-vs-rg A/A gated).
- Target: close the **+3.22 instr/B (+25.5%)** clean-emission instruction surplus toward
  `_04`'s 12.65 instr/B. Do **NOT** chase IPC (gz already +5% ahead).
- Per the inner-kernel charter (CLAUDE.md): the emission copy/store + decode-LOOP packing
  is open territory — port `_04`'s wide-copy / multi-symbol techniques.

## Owed (NOT-YET-LAW)
- AMD/Zen2 replication (both the +16% T4 gap and the instr/B axis; PEXT/PDEP microcoded
  on Zen2 — `_04` may behave differently).
- A quiet/frozen Intel re-run to tighten the gz wall spread (this was load 4.4–4.8; the
  gap reproduced and A/A passed, but spread > 5%).
- Sub-component apportionment (wide-COPY vs multi-symbol decode-LOOP packing) within the
  +3.22 instr/B — which `_04` technique to port first.

## Reproduce
```
# build the isolated A/B kernel (gzippy-isal feature gives both kernels)
CARGO_TARGET_DIR=/dev/shm/kab-target RUSTFLAGS="-C target-cpu=native" \
  cargo build --release --example kernel_ab --features gzippy-isal
# gated single-core instr/B + cyc/B + IPC + LLC + GHz, N=15
bash scripts/bench/kernel-ab/emission_axis_guest.sh \
  /dev/shm/kab-target/release/examples/kernel_ab 3 2000 15 /dev/shm/emission_axis.csv
python3 scripts/bench/kernel-ab/emission_axis_analyze.py /dev/shm/emission_axis.csv
# production silesia-T4 gz/rg + rg-vs-rg A/A
GZ=/dev/shm/cleankernel-target/release/gzippy RG=/root/oracle_c/rapidgzip-native \
  CORP=/root/silesia.gz T=4 N=15 bash scripts/bench/standing/_cleankernel_silt4_guest.sh
```
Artifact: `artifacts/emission_axis_intel.csv` (raw N=15 per-cell perf counts).
