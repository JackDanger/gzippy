# Our L1 port is EXACT, and hash3 is the entire remaining gap — measured

Measured 2026-08-01, local M1, `cargo test --release --lib l1_bakeoff`, first
256 KB of each file, deterministic and load-immune. Two builds, by hand — no
script between the code and the number.

## With the length-3 table DISABLED, we match libdeflate exactly

`HT_MAX_LEN3_OFFSET = 0`:

    file            ht_fast   libdeflate   delta
    armexe.elf       122402      122402       0
    data.parquet      82499       82499       0
    data.csv          40374       40374       0
    engine.wasm      121364      121364       0
    minjs.min.js      92717       92717       0
    data.json         34517       34517       0
    aozora.txt       102661      102661       0
    dickens          107483      107483       0
    TOTAL            704017      704017       0     ht_pct = +0.000

**Identical compressed size on all eight files**, across text, JSON, CSV,
parquet, minified JS, an ELF binary and a wasm module. `ht_fast` minus hash3 IS
libdeflate's `deflate_compress_fastest` + `ht_matchfinder`. The transliteration
is exact, and this is the cleanest confirmation of that we have ever had.

(Size-identical, not proven byte-identical — that needs a sha compare. But eight
files of eight types agreeing to the byte count is not coincidence.)

## So hash3's effect is measurable against an EXACT baseline

Turning it back on (`HT_MAX_LEN3_OFFSET = 4096`) moves every file, and the sign
splits perfectly by content:

    hash3 GAINS on binaries          hash3 COSTS on text
      armexe.elf     -4013             dickens        +644
      data.parquet    -984             aozora.txt     +550
      data.csv        -210             data.json      +449
                                       minjs.min.js   +313
      subtotal       -5207             engine.wasm     +61
                                       subtotal      +2017

Net −3190 across the eight, which is the whole of `ht_fast`'s −0.453% vs
libdeflate. **The hypothesis is confirmed exactly: the length-3 table earns
bytes on binaries and costs them on text.**

## What this changes

The choice is no longer "our L1 vs libdeflate's". It is:

- **hash3 OFF** → we TIE libdeflate on all 8. Zero losses, zero wins. Every L1
  cell becomes a byte-tie, which per `project_t4_seam_is_a_step_function` is a
  zero-headroom state — safe but permanently unwinnable.
- **hash3 ON at 4096** → we WIN 3 and LOSE 5.

Neither is the goal. The goal is per-label non-worse on all of them, and the
measurement says the two effects are cleanly separable by CONTENT, which we may
not detect (non-negotiable #3 forbids a content detector choosing a parser).

**But they may also be separable by OFFSET, which is not detection — it is a
static bit-cost rule, exactly what gzip's TOO_FAR already is.** A length-3 match
at a large offset costs more bits than three literals; the question is whether
the offset at which that flips differs between our win files and our loss files.
The untested values are 512 / 1024 / 2048 / 8192 / 32768. If some offset keeps
most of the −5207 while shedding most of the +2017, it closes L1 outright.

## ⛔ A CORRECTION I OWE THE SWEEP

The broken sweep reported `+0` for every delta at `len3_off=0`, and I dismissed
it as obviously wrong — *"would mean we tie libdeflate exactly on all eight
files"*. **That was the true measurement.** I called a correct result implausible
because I did not believe our port could be exact.

The sweep's OTHER symptom — identical totals at both settings, both equal to the
`Fast` figure — was a genuine parsing bug, and it is what made the whole output
look untrustworthy. One real defect next to one correct-but-surprising number,
and I discarded both.

**Implausible-looking output deserves the same treatment as implausibly-good
output: check it, do not assume it.** The campaign already knows
"implausibly-good is a provenance alarm"; this is the mirror case, and it cost a
confirmed hypothesis several hours.

## Scope

- 256 KB blocks, L1 only, size only, one box. Says NOTHING about the wall, which
  is what killed this lever at T1 before.
- `ht_fast` is NOT routed in production; the shipped L1 is `Strategy::Fast`, at
  +1.634% vs libdeflate. See `docs/board/l1-next-lever.md` for the routing
  blocker (#227's `params_parallel`, since it must be gated T>1).

---

# ⭐ MEASURED: offset 256 STRICTLY DOMINATES 4096 on the per-label bar

The separation the section above predicted exists, and it is large. Same test,
same 256 KB blocks, `HT_MAX_LEN3_OFFSET` swept by hand (one build per value,
each written to its own file):

| file | off=0 | **off=256** | off=4096 (shipped default) |
|---|---|---|---|
| armexe.elf | 0 | **−2177** | −4013 |
| data.parquet | 0 | **−428** | −984 |
| data.csv | 0 | **−253** | −210 |
| engine.wasm | 0 | **0** | +61 |
| minjs.min.js | 0 | **−274** | +313 |
| data.json | 0 | **+102** | +449 |
| aozora.txt | 0 | **+163** | +550 |
| dickens | 0 | **+232** | +644 |

    binaries (3):  off=256  −2858   vs  off=4096  −5207   (keeps 55% of the gain)
    text     (5):  off=256   +223   vs  off=4096  +2017   (sheds 89% of the cost)

**Per-file, which is the only thing that grades:**

    off=4096   win 3, lose 5
    off=256    win 4, TIE 1, lose 3

Two files flip outright — `minjs.min.js` +313 → **−274**, `engine.wasm` +61 → an
exact tie — and `data.csv` is *better* at 256 (−253) than at 4096 (−210). The
three remaining losses shrink from +449/+550/+644 to **+102/+163/+232**.

## Why this is principled, not fitted

`HT_MAX_LEN3_OFFSET` is gzip's TOO_FAR rule: *a length-3 match beyond this
offset costs more bits than three literals*. That is a BIT-COST statement, and
bit cost depends on the Huffman code — which differs sharply between text and
binary. 4096 is gzip's constant, inherited unexamined. The measurement says the
tradeoff for THIS matchfinder flips much earlier.

It is also a STATIC RULE, not a content detector: nothing inspects the data to
choose it. Non-negotiable #3 is not engaged.

## What is NOT claimed

- **256 is not established as the optimum.** It was the SMALLEST value probed;
  the trend suggests lower may be better still for text. 64 and 128 are untested,
  and 512/1024/2048 were still building when this was written.
- 256 KB blocks, L1 only, SIZE only, 8 files, one box. Says nothing about the
  wall — and the wall is what killed this lever at T1 before. A smaller offset
  should if anything be CHEAPER (fewer length-3 candidates accepted), but that is
  an argument, not a measurement.
- `ht_fast` is still NOT routed. The shipped L1 is `Strategy::Fast` at +1.634%.
  Routing is blocked on #227's `params_parallel` since it must be gated T>1.
- The corpus here is the 8-file bakeoff set, not the 22-file board.
