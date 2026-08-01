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
