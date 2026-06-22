# SCAFFOLD-LOCATE results (MEASUREMENT 3) — where the gz-contig-driver scaffold cost lives

Branch `kernel-converge-A`. Instrument: `examples/streaming_thin.rs` cheap-arm
removal-oracle ablations + `scripts/bench/_scaffold_locate_guest.sh`. ALL arms run
the SAME isal binary; inner decode (igzip `read_header` + `_04`/`_base`) held
CONSTANT. `cheap` = gz contig driver baseline (== `igzipbarecheap`). Each `cheap_*`
REMOVES one driver sub-region; `(cheap-cheap_X)/cheap` BOUNDS its share. `igzip` =
ISA-L monolith (WITH CRC) = the x86 bar. CLEAN SCAFFOLD = `(cheap-igzip)/igzip`.

Gate-0: every arm bytes==zcat (PASS all cells). Per-region NON-INERT proof (stderr
`ABLATE` line, confirmed): nosink sink_calls=N but 0 writes; nozero/big uninit cap;
noin input_copied=0; big memmove_calls=0 + cap=whole-output. A/A `|cheap-cheap2|`
reported per cell (≪ inter-arm Δ).

Sub-region share sign: **positive = removing X SPEEDS UP** (X is a real driver cost);
**negative = removing X SLOWS DOWN** (X is net-beneficial — do NOT remove).

## INTEL (neurotic i7-13700T LXC, taskset cpu4, best-of-15, load ~2-3)

| corpus   | cheap ms | igzip ms | SCAFFOLD | noin (input-copy) | nozero | nosink | big (memmove-rm) | cheap_noin vs igzip |
|----------|----------|----------|----------|-------------------|--------|--------|------------------|---------------------|
| silesia  | 708.3    | 658.0    | **7.65%**| **+9.70%** (68.7ms)| 0.17% | 0.05% | **−20.93%**      | 639.6 < 658 (gz WINS)|
| nasa     | 236.2    | 232.6    | **1.54%**| **+8.53%** (20.1ms)| 0.02% | 0.25% | **−58.87%**      | 216.0 < 232.6 (gz WINS)|
| monorepo | 112.0    | 105.8    | **5.94%**| **+7.82%** (8.8ms) | 0.08% | −0.14%| **−29.26%**      | 103.3 < 105.8 (gz WINS)|
| squishy  | 1426.1   | 1288.5   | **10.68%**| **+11.94%** (170ms)| −0.03%| 0.11% | **−21.25%**      | 1255.8 < 1288.5 (gz WINS)|

A/A `|cheap-cheap2|`: silesia 2.30 / nasa 0.32 / monorepo 0.01 / squishy 0.27 ms — all
≪ the noin/big Δ. GATE0(bytes)=PASS every cell.

### INTEL VERDICT (gated, removal-oracle)
1. **INPUT `.to_vec()` COPY is the SOLE scaffold sub-region.** Removing it (cheap_noin,
   byte-exact) speeds the gz driver 7.8–11.9% on EVERY corpus, and on every corpus that
   single removal EXCEEDS the entire measured CLEAN SCAFFOLD — so with the input copy
   gone the gz contig driver **BEATS the igzip monolith on all 4 corpora** (and gz still
   skips CRC, so this is conservative). ⇒ the "clean scaffold" the prior decomposition
   banked is, on Intel, ENTIRELY the cheap arm's input copy (a cost igzip never pays —
   isal_inflate reads the file in place).
2. **Buffer zeroing (nozero) and output sink (nosink) are TIEs** (≤0.25%, within A/A
   spread). Not levers.
3. **Window memmove + per-batch flush (big) is NET-BENEFICIAL.** Removing it (one giant
   buffer) is 21–59% SLOWER — the small reused ~12 MiB contig buffer + 32 KiB
   memmove-retain is a cache-locality WIN. The contig design is CORRECT; do NOT "converge"
   it away.

## AMD (solvency EPYC 7282 Zen2, bare-metal FROZEN gov=performance boost=0, taskset cpu4, best-of-15)

Box thawed + verified after: gov=ondemand / boost=1 / paranoid=4.

| corpus   | cheap ms | igzip ms | SCAFFOLD | noin (input-copy) | nozero | nosink | big (memmove-rm) | cheap_noin vs igzip |
|----------|----------|----------|----------|-------------------|--------|--------|------------------|---------------------|
| silesia  | 401.7    | 357.7    | **12.32%**| **+15.43%** (62.0ms)| 0.46% | −0.28%| **−42.04%**      | 339.7 < 357.7 (gz WINS)|
| nasa     | 132.4    | 131.8    | **0.46%**| **+14.44%** (19.1ms)| 0.19% | 0.23% | **−113.34%**     | 113.3 < 131.8 (gz WINS)|
| monorepo | 63.5     | 57.9     | **9.74%**| **+12.68%** (8.0ms)| 0.60% | 0.82% | **−53.85%**      | 55.4 < 57.9 (gz WINS)|
| squishy  | 815.3    | 704.4    | **15.75%**| **+18.06%** (147ms)| 0.07% | −0.18%| **−37.89%**      | 668.1 < 704.4 (gz WINS)|

A/A `|cheap-cheap2|` (min-to-min): silesia 1.85 / nasa 0.06 / monorepo 0.30 / squishy 3.61 ms
— all ≪ the noin Δ. (Per-arm best-worst spread is inflated by load tails on silesia/squishy;
the min-to-min A/A is the correct significance estimator and is tiny.) GATE0(bytes)=PASS all.

## CROSS-ARCH VERDICT (gated removal-oracle; Intel AND AMD AGREE → LAW-grade)

1. **The input `.to_vec()` copy is the SOLE scaffold sub-region — on BOTH arches.**
   Removing it speeds the gz driver 7.8–11.9% (Intel) / 12.7–18.1% (AMD), and on
   ALL 8 cells (4 corpora × 2 arch) that single removal makes the gz contig driver
   **BEAT the igzip monolith** (gz also skips CRC → conservative). nasa is the tell:
   SCAFFOLD ≈ 0 (1.5% Intel / 0.5% AMD) yet noin removes ~14% — i.e. cheap already
   ≈ igzip on nasa *because* nasa's input is small relative to output, so the copy is
   cheap there; where the input is larger (squishy/silesia) the copy dominates the
   apparent "scaffold."
2. **Buffer zeroing (nozero) and output sink (nosink) are TIEs** (≤0.8%, ≈ A/A) on
   both arches. Not levers.
3. **Window memmove + per-batch flush (big) is strongly NET-BENEFICIAL** on both arches
   (removing it is 21–59% slower Intel / 38–113% slower AMD). The reused small contig
   buffer + 32 KiB memmove-retain is a cache/TLB-locality WIN. The contig design is
   CORRECT — do NOT converge it toward a whole-output buffer.

## WHAT THE "CLEAN SCAFFOLD" ACTUALLY IS (reframes the prior decomposition)

The prior decomposition (`project_x86_decomposition_2026_06_21`) banked CLEAN SCAFFOLD =
(cheap−igzip)/igzip = 6–16% as "the cross-arch-consistent gz-contig-driver-vs-monolith
overhead" and the recommended convergence target. This LOCATE shows that gap is, on BOTH
arches, the **cheap-arm's input `.to_vec()` memcpy** — a cost that exists ONLY in the
measurement harness (the cheap/igzip_bare arms copy `file_bytes[hdr..].to_vec()+64` to
hand the isal kernel a `*mut` + read-ahead pad), which **igzip never pays** (isal_inflate
reads `next_in` in place) and which **gz PRODUCTION does not pay either**:
`decompress_parallel(gzip_data: &[u8], …)` (src/decompress/parallel/single_member.rs:175)
takes the input as a BORROWED slice and the bit-reader reads it in place — no whole-input
copy. *(Production-borrow = HYPOTHESIS-tier code-read; the gated fact is the cheap_noin
removal-oracle.)*

With the input copy excluded, the INTRINSIC gz-contig-driver structural overhead vs the
igzip monolith is ZERO-or-NEGATIVE on all 8 cells (gz wins). ⇒ **There is no scaffold to
converge.** The contig driver shape (borrow input + decode into a small reused contig
buffer + 32 KiB memmove-retain + per-batch flush) is at-or-faster than the igzip monolith
holding the inner decode constant.

## RECOMMENDED CONVERGENCE TARGET

**Do NOT start a driver-convergence rewrite toward the igzip monolith.** The removal-oracle
shows the gz contig driver already meets-or-beats the monolith once the harness-only input
copy is excluded; the apparent "scaffold lever" was a harness artifact. Concretely:

- **Next gate (prerequisite to any rewrite):** a CLEAN production-path race — the real
  parallel-SM T1 driver (which borrows input) vs igzip, holding the kernel constant — to
  confirm production carries no input-copy and is at/above driver parity. (The cheap arm is
  a harness proxy that ADDED a copy; it is not the production driver.)
- **The residual gz-native T1 deficit is the KERNEL, not the scaffold.** Per the kernel
  decomposition that is Intel-specific & corpus-dependent (null/negative on Zen2). So the
  driver is NOT the place to spend a rewrite; if any T1 work is funded it is the inner
  kernel (Intel-only ROI), which the prior cross-arch verdict already priced as poor ROI.
- **Keep the contig+memmove design** (gated net-beneficial); do not "faithfully converge"
  it to a monolith circular-window/whole-output buffer — that REGRESSES 21–113%.
