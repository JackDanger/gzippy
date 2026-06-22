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

## AMD (solvency EPYC 7282 Zen2) — PENDING
