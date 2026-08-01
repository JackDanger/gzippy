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

---

# ⛔ CORRECTION 3 — "36.8% of the excess" compared our module against ZERO. It is <= 12.0%.

The previous section reported `block_split.rs` = 33,152,259 Ir = **36.8% of the
entire T1 instruction excess**. That divided OUR module total by the excess, as
if libdeflate paid nothing for block splitting. It pays plenty — its
`observe_literal`/`observe_match`/`ready_to_check_block` are `forceinline` and
land inside `deflate_compress_greedy`, which is exactly why they never appeared
as a separate line and why the earlier note said "no counterpart in their
top-99%". That note was about their PROFILE LAYOUT and I then used it as if it
were about their COST.

Measured per source line on their side:

| libdeflate block-splitting | Ir |
|---|---|
| `observe_literal` | 5,244,687 |
| `observe_match` | 11,298,612 |
| `ready_to_check_block` | 5,528,392 |
| `merge_new_observations` | 150,030 |
| `do_end_block_check` (attributed part) | 99,560 |
| **total (a LOWER bound — see caveat)** | **22,321,281** |

    ours   33,152,259 Ir   (own file, fully attributed)
    theirs 22,321,281 Ir   (inlined, attributed lines only)
    gap    <= 10,830,978 Ir  =  <= 12.0% of the 90,030,697 T1 excess

⚠ Their figure is a LOWER bound and the gap an UPPER bound: their block-split
code is inlined into `deflate_compress_greedy`, so any unattributed remainder
falls into that function's bucket and cannot be separated. Our own module
carries 4,616,982 Ir of "unidentified lines" that ARE visible precisely because
it is a separate file. Comparing a fully-attributed module against a partially-
attributed inlined one biases the gap UPWARD, which is the direction that
flatters the finding.

## What the per-line diff DID establish — and it is exact

| | ours | libdeflate | delta |
|---|---|---|---|
| `observe_literal` | 5,244,687 | 5,244,687 | **0 — identical to the instruction** |
| `observe_match` | 15,064,816 | 11,298,612 | **+3,766,204 = exactly +2.0 Ir/match** |

1,883,102 matches x 2 Ir = 3,766,204 exactly. The literal path is
instruction-for-instruction equal, so this is NOT generic Rust-vs-C overhead —
it is two specific instructions in `observe_match`, and it localises further:
`num_new_observations += 1` costs **4 Ir in our match path, 1 Ir in theirs**
(their compiler keeps the field in a register across the inlined call; ours
reloads it), while our index computation is 1 Ir cheaper. Net +2.

The other named item is `ready_to_check_block`: ours 7,989,750 Ir on its first
condition alone (3.04 Ir per token, called per position) against their 5,528,392
for all three conditions.

## Full accounting of our module (sums to the total, exactly)

| | Ir | share |
|---|---|---|
| observers | 20,309,503 | 61.3% |
| `ready_to_check_block` first condition | 7,989,750 | 24.1% |
| unidentified lines | 4,616,982 | 13.9% |
| `do_end_block_check` (the real work) | 236,024 | 0.7% |
| **total** | **33,152,259** | 100% |

The actual split DECISION is 0.7% of the module. Ninety-nine percent of block
splitting's cost is the per-position bookkeeping that feeds it.

## Standing after three corrections

- matchfinder at T1: **2.20% CHEAPER than theirs**. Unchanged, still the
  strongest result of the day, still kills matchfinder inner-loop levers.
- non-matchfinder half: **+39.4%**, +100,969,707 Ir, 112.2% of the excess.
  Unchanged.
- block splitting's contribution to that: **<= 12.0%**, not 36.8%.
- The remaining ~88% of the excess is still UNATTRIBUTED to a named component.
  `parse/mod.rs` (71.8M), `emit_sequences` (51.9M), `bitstream.rs` (34.1M),
  `tables.rs` (28.2M) and the `core::ptr`/`uint_macros` rows are where it must
  be, and none of them has been diffed against the vendor line-by-line yet.
  **That is the next measurement.**
