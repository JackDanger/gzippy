# 93.7% of our instruction excess at L2 is OUTSIDE the matchfinder

Measured 2026-08-01 on **trainer** (Intel i7-13700T, the only box with
valgrind), callgrind + `callgrind_annotate`, dickens (12.17 MB), byte-identical
output on both arms at both levels.

- ours: `/root/gzippy/target/release/gzippy`, main `120bfa9c`, built
  `CARGO_PROFILE_RELEASE_STRIP=false CARGO_PROFILE_RELEASE_DEBUG=true`.
  Verified codegen-equivalent to the shipping build: compressed output sha
  `d608c061737bce1c` from both the stripped and the unstripped binary.
- rival: `vendor/libdeflate` @ `22f6e023` + the probe patch, built
  `RelWithDebInfo -g`. Verified byte-identical to the
  `/opt/homebrew/bin/libdeflate-gzip` every board number was measured against.
- Totals are callgrind's own `summary:` line, not a parser's sum. **This
  matters**: `fulcrum why` layer [2] reported these same two runs as
  10,307,929,423 vs 5,987,049,889 (ratio 1.72) by double-counting call-arc
  inclusive costs — an 11.6x/7.95x ASYMMETRIC inflation. Fixed in fulcrum
  PR #18; the numbers below come from callgrind directly.

## L2 — greedy(6,10), the coordinate where our cells FAIL

    ours   886,667,693 Ir
    theirs 752,825,508 Ir      ours is +17.8%

| component | ours | libdeflate | delta |
|---|---|---|---|
| **matchfinder** (chain walk + hash/insert + its SIMD) | 504,948,700 | 496,567,178 | **+1.69%** |
| **everything else** | 381,718,993 | 256,258,330 | **+49.0%** |

    total excess            +133,842,185 Ir
      inside the matchfinder  +8,381,522   6.3% of the excess
      outside it            +125,460,663  93.7% of the excess

The matchfinder is at PARITY. Ninety-four percent of the excess is elsewhere.

### Two named components with no counterpart in their top-99%

| | Ir | % of our arm | % of the excess |
|---|---|---|---|
| `block_split.rs` (inlined into `greedy::run_resumable`) | 33,022,690 | 3.7% | **24.7%** |
| `__memcpy_avx_unaligned_erms` (libc) | 18,322,678 | 2.1% | **13.7%** |

Together **38.4% of the entire instruction excess**.

⚠ "No counterpart in their top-99%" is a statement about their PROFILE, not
about their algorithm. libdeflate has block-splitting logic
(`deflate_should_end_block`) which is very likely inlined into
`deflate_compress.c:deflate_compress_greedy` (126.7M) and would not appear as
its own line. What IS established is that we spend 33.0M Ir in a separate
block-splitting module and 18.3M Ir in libc `memcpy`, and that their whole
non-matchfinder cost is 256.3M against our 381.7M.

## L9 — lazy2(600, MAX), the coordinate where we WIN

    ours   2,548,687,249 Ir
    theirs 1,953,958,407 Ir     ours is +30.4%

| | L2 | L9 | change |
|---|---|---|---|
| our total Ir | 886.7M | 2,548.7M | **2.9x** |
| `block_split.rs` | 33.0M (3.7%) | 28.2M (1.11%) | **-15%** |
| `memcpy` | 18.3M (2.1%) | 18.0M (0.71%) | **-1.7%** |

**Both are FLAT in absolute Ir while everything around them tripled.** They are
input-driven, not depth-driven. That is a depth-independent per-pass cost,
measured on Intel with an instruction counter — the same SHAPE the M1 wall
model found as `F` (`docs/board/fixed-vs-pernode.md`), on a different box, a
different arch, and a different instrument.

This was a PREDICTION, registered before the L9 run: "if block_split and memcpy
are input-driven fixed costs, their absolute Ir should barely move at depth
600." It held on both.

## ⚠ CORRECTION — instructions and wall DIVERGE here, and the M1 record needs it

`docs/board/fixed-vs-pernode.md` concluded from M1 wall timings that "our chain
walk is 17.3% CHEAPER per node." At L9 on trainer:

    ours   hc.rs:lazy::run_resumable            1,866,549,864 Ir
    theirs hc_matchfinder.h:deflate_compress_lazy2 1,480,691,091 Ir   ours +26.1%

**We execute 26% MORE instructions in the matchfinder and still win the wall by
12.3%.** So "cheaper per node" is true in MILLISECONDS and false in
INSTRUCTIONS: we issue more and stall less. Both statements are correct; they
are about different quantities. This is the campaign's own standing rule
("instruction counts LOCATE, they never predict the wall") landing on my own
result — and it is consistent with the banked
`project_encoder_deficit_is_loads_not_stalls`, which records that we beat
libdeflate on IPC, stalls, cache and branches.

## What is NOT claimed

- **That the 51M Ir of block_split+memcpy CAUSES the 8.5 ms `F` gap.** Two
  instruments independently locate a depth-independent per-pass cost, and one
  of them names two components. Connecting them quantitatively would need the
  cycles counter, and layer [3] SKIPPED (its `gzip` oracle exited 1, and the
  trainer kernel dropped every requested event). Ir does not convert to ms —
  see the correction above, which is exactly that mistake caught in the act.
- **That removing block splitting is a win.** It is a size mechanism; deleting
  it would change output bytes and is subject to clause 3. Nothing here has
  been ablated.
- **Scope: dickens only, trainer/Intel only, L2 and L9 only, T1 only.** The M1
  numbers are a different arch. Nothing generalises to another file until
  measured (hard stop #3).
- `deflate_into`'s own hc.rs/common.rs rows (14.4M + 2.6M) were folded into the
  matchfinder totals above; they are the resumable driver, not a third
  component.

## What this kills, and what it opens

**KILLS:** any lever aimed at the matchfinder inner loop at L2. It is at
+1.69% parity in instructions and 6.3% of the excess. Three prior levers aimed
there.

**OPENS:** the non-matchfinder half, which is +49.0% at the exact coordinate
where every failing wall cell lives. `block_split.rs` and the libc `memcpy` are
the two largest NAMED, depth-independent items in it. Neither has been ablated,
and the next step is `fulcrum ab ablate`, not an edit.

---

# ⛔ CORRECTION (same day) — the profiles above were T16, not T1. Two claims change.

`gzippy -2 -c FILE` with **no `-p` flag uses `num_cpus`**. Trainer has 16. Every
callgrind run in the sections above was therefore the **T16** path, while the
rival (`libdeflate-gzip`) is single-threaded by construction. The failing wall
cells are at **T1**. `scripts/campaign/board-wall.sh:99` carries this exact
warning — *"without an explicit -pN gzippy uses num_cpus, and an agent once
measured T10 believing it was T1"* — and I hand-rolled the callgrind invocation
without it.

Re-measured with an explicit `-p1`, same binary, same input, same level:

| | T16 (what was profiled) | **T1 (the failing coordinate)** |
|---|---|---|
| our total Ir | 886,667,693 | **842,856,205** |
| `__memcpy_avx_unaligned_erms` | 18,322,678 (2.07%) | **261,274 (0.03%)** |
| `block_split.rs` | 33,022,690 (3.72%) | **33,152,259 (3.93%)** |

## RETRACTED: the memcpy finding

18.3M Ir of libc `memcpy`, and the size-scaling test that "confirmed" it as one
full copy of the input (predicted 1.78M on armexe.elf, observed 1,897,836), were
**both T16 measurements**. At T1 the figure is 261,274 Ir — 70x smaller, 0.03%
of the arm, not a finding.

The copy is real, but it is the **T>1 BUF_PAD input copy already recorded in
commit #210** ("T>1 copies the entire input for BUF_PAD it already has —
verified, not built"). I re-derived a known T>1 result, mislabelled it T1, and
size-scaled it into a confirmation. The size-scaling test was valid and its
prediction held — it just tested a hypothesis at the wrong coordinate. **A
correct prediction about the wrong coordinate is not evidence for the claim it
was made about.**

## SURVIVES AND STRENGTHENS: block_split, and the matchfinder verdict

`block_split.rs` barely moves between T16 and T1 (33.02M -> 33.15M), so it is a
genuine T1 cost. Against the correct T1 total the picture sharpens:

    ours   842,856,205 Ir     theirs 752,825,508 Ir     ours +11.96%

| component | ours | libdeflate | delta |
|---|---|---|---|
| matchfinder (walk + hash/insert + SIMD) | 485,628,168 | 496,567,178 | **-2.20%** |
| everything else | 357,228,037 | 256,258,330 | **+39.4%** |

    total excess             +90,030,697 Ir
      inside the matchfinder  -10,939,010   (-12.2%)
      outside it             +100,969,707   (112.2%)

At T1 our matchfinder is not merely at parity — it is **2.20% CHEAPER in
instructions**, so it *offsets* part of the excess, and the non-matchfinder half
must carry 112.2% of it.

**`block_split.rs` alone is 33,152,259 Ir = 36.8% of the ENTIRE T1 instruction
excess**, at the exact level and thread count where every failing wall cell
lives.

## What this does NOT change

- The M1 wall measurements (`fulcrum ab paired`) all passed `-p1` explicitly and
  are unaffected.
- The matchfinder conclusion is unaffected in direction and stronger in
  magnitude.
- The L9 comparison above (ours +26.1% Ir in the matchfinder while winning the
  wall) was also T16 on our side; the T1 re-measurement is recorded below it.

## The depth-independence claim, re-checked at T1

    L2 T1:  total   842,856,205    block_split 33,152,259 (3.93%)   memcpy 261,274 (0.03%)
    L9 T1:  total 2,505,128,095    block_split 28,377,310 (1.13%)   memcpy 349,161 (0.01%)

Total Ir grows **2.97x** from depth 6 to depth 600; `block_split.rs` falls 14.4%
in absolute terms. It is input-driven, not depth-driven — the T16 result held
after the coordinate was fixed, and it is now measured where the cells fail.
`memcpy` is negligible at T1 at BOTH levels, which is the same retraction seen
from the other end.

So the depth-independent per-pass cost is real and `block_split.rs` carries it;
the `memcpy` half of the earlier claim does not exist at T1.
