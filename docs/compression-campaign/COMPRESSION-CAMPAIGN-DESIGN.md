# gzippy COMPRESSION CAMPAIGN — pre-staged design

Status: DESIGN (pre-build), Mac-only, no box access, no commits. Author: design-leader
session 2026-07-05. Deliverables: this doc + `compression-spec-draft.json` +
`dominance-spec-decomp.json` + `dominance-spec-comp.json` (all in scratchpad).

Governing law: the anti-bias preamble + measurement protocol in CLAUDE.md apply from
birth. Nothing here is a finding; every claim about a lever/front below is labelled
`HYPOTHESIS (unvalidated)` and carries the exact measurement that would test it. The
purpose of this doc is to make compression **day-1 a baseline run, not a tooling
build**: the scoreboard, oracle, field, corpus, and finish-line specs already exist
the moment the decompression gap closes.

This design **extends, does not fork**, the `fulcrum scoreboard` artifact model
(`fulcrum-mac/docs/scoreboard-design.md`, `src/scoreboard.rs`) and the `fulcrum run`
FieldBaseline spec shape (`examples/first-run-baseline.spec.json`). The only new
machinery is (a) a **second measured axis (compressed bytes)** on each cell and (b) a
`--dominance` finish-line checker for `scoreboard diff`. Both are additive.

---

## 0. Why compression is NOT decompression-with-a-different-verb

Decompression has ONE measured axis: wall (output bytes are fixed — the decoded
plaintext, identical for every tool). A cell verdict is a 1-D significance test on
wall; "gzippy strictly fastest" is well-defined.

Compression has TWO measured axes that TRADE OFF against each other:

- **wall** (noisy — needs the full A/A + TOST significance machinery), and
- **compressed size in bytes** (EXACT — deterministic integer, zero noise, no A/A).

A faster compressor that emits a larger file has **not** won — it bought speed with
ratio. The whole campaign verdict for compression must therefore be a **2-D Pareto**
statement, not a wall race. This is the single most important design decision and the
one most likely to poison the campaign if gotten wrong (hence the adversarial review in
§7 is aimed squarely at it).

gzippy's own architecture makes this unavoidable, not academic: its parallel path
(`ParallelGzEncoder`, T>1 L1–5/L10–12) compresses **independent blocks with the
dictionary reset at every block boundary** (`src/compress/parallel.rs:15`,
`get_block_size_for_level`), which is *faster and parallel-decompressible but produces
a strictly larger file* than a dictionary-continuing encoder. Its pipelined path
(`PipelinedGzEncoder`, T>1 L6–9) shares the previous block's INPUT as a dictionary
(`src/compress/pipelined.rs:1–15`) to *recover pigz's ratio at the cost of
sequential-only decode*. So gzippy deliberately sits at different points on the
speed/ratio plane depending on (level, T). A 1-D wall verdict would score the
independent-block path as a "win" while it is silently shipping bigger files. The
verdict machinery must see both axes or it will certify ratio regressions as speed wins.

---

## 1. 2-D PARETO CELL SEMANTICS (the load-bearing section)

### 1.1 Coordinates

A **compression cell** is `(box × corpus × T × level-class)`. Each cell produces, per
tool, a point `P = (wall, bytes)` where `wall` = median wall (seconds, box-local
monotonic timing per the scoreboard exec model) and `bytes` = size of the compressed
output (exact, from the correctness arm's output — see §2). gzippy is the *subject*;
every other tool is a *rival* producing its own point(s) in the cell.

### 1.2 The two axes are certified differently

- **wall axis** — inherits the scoreboard's full significance machinery verbatim:
  interleaved best-of-N, A/A arm, paired sign test, TOST tie-certification,
  `spread`-gated `|ratio−1|`. Output of the wall comparison of two points is one of
  `{FASTER, SLOWER, WALL-TIE(TOST), WALL-VOID}`.
- **bytes axis** — EXACT. `bytes_g` vs `bytes_i` is an integer comparison, no A/A, no
  spread, no TOST. Output is one of `{SMALLER, LARGER, BYTE-EQUAL}`. A 0.1%-larger
  output is a byte LOSS, full stop. (Determinism of `bytes_g` per cell is asserted in
  §2.5 — it must not wobble run-to-run or the "exact" claim is a lie.)

Combining a noisy axis with an exact axis is the subtlety. The rule: **the bytes axis
is never softened by wall noise and the wall axis is never hardened by byte exactness.**
Each axis keeps its own certification; the 2-D verdict is a function of the pair.

### 1.3 Pairwise Pareto verdict (gzippy point G vs one rival point R)

Let `wallcmp ∈ {FASTER, SLOWER, WALL-TIE, WALL-VOID}` (G relative to R) and
`bytecmp ∈ {SMALLER, LARGER, BYTE-EQUAL}` (G relative to R).

If `wallcmp == WALL-VOID` → the pair is `VOID` (apparatus failed; not a verdict — same
as scoreboard today).

Otherwise map "no worse" / "strictly better" per axis:

- wall no-worse = `{FASTER, WALL-TIE}`; wall strictly-better = `{FASTER}`.
- byte no-worse = `{SMALLER, BYTE-EQUAL}`; byte strictly-better = `{SMALLER}`.

Then:

| condition | pairwise verdict |
|---|---|
| G no-worse on BOTH axes AND strictly-better on ≥1 | **G-DOMINATES** |
| G no-worse on both, strict on neither (tie+equal) | **MATCH (2-D TIE)** |
| R no-worse on BOTH axes AND strictly-better on ≥1 | **R-DOMINATES (LOSS)** |
| each axis favors a different tool (faster-but-larger, or slower-but-smaller) | **CONTESTED (Pareto-incomparable)** |

This is exactly the brief's "WINS iff Pareto-dominates or matches; LOSES iff strictly
dominated; else contested/TIE" — made mechanical.

### 1.4 Cell verdict = worst-case over the rival field in the level-class

A cell has ONE gzippy point G (its best point in the level-class — see §1.6) and a set
of rival points `{R_j}` (every rival's point whose ratio lands in the same class band).
Aggregate G against each `R_j`, then reduce **worst-first** (a campaign whose finish
line is "gzippy is never beaten" must fail a cell if ANY rival beats it):

- any `R_j` gives `LOSS` → cell = **LOSS** (record the dominating rival).
- else any `CONTESTED` → cell = **CONTESTED** (gzippy on the frontier but a rival offers
  a different trade — record which and on which axis gzippy is behind).
- else all `G-DOMINATES`/`MATCH`, at least one `G-DOMINATES` → cell = **WIN**.
- else all `MATCH` → cell = **TIE**.
- any pair `VOID` and no LOSS yet → cell = **VOID** (can't certify; re-measure).

### 1.5 Two finish-line tiers (the achievable vs the aspirational)

A 2-D problem has no single "strictly best point" — the frontier is a *curve*. So the
campaign uses two tiers, both recorded per cell:

1. **HULL-MEMBER (the floor / true finish line).** gzippy is *not strictly dominated by
   any rival point anywhere in the corpus* (across ALL rival levels, not just the
   class). Equivalently: gzippy's (wall, bytes) point lies on-or-above the field's 2-D
   Pareto hull. This is the honest "gzippy is never a dumb choice" bar and it IS
   achievable. LOSS ⇒ not a hull member.
2. **BAND-WIN (the stretch / per-class dominance).** Within a ratio band, gzippy
   `G-DOMINATES`-or-`MATCH`es every rival — i.e. at gzippy's compression ratio it is the
   fastest tool AND no rival is simultaneously smaller and faster. This is the
   compression analogue of decompression's "strictly fastest."

The campaign target: **HULL-MEMBER at every cell (mandatory), BAND-WIN at every cell
(aspirational, VoI-ordered but never dropped — infinite-funding rule).** A CONTESTED
cell is acceptable **iff** it is still a HULL-MEMBER (gzippy trades but is not
dominated); it is a genuine miss only when it drops off the hull (LOSS).

### 1.6 Level-CLASS mapping — anchored by MEASURED ratio band, not nominal level

> **AMENDED by review disposition D4 (§7): the partition below is SUPERSEDED by
> NESTED QUALITY-TARGET CONSTRAINTS.** Each class is a target `r ≤ r(anchor)` (a
> quality floor), nested C3⊂C2⊂C1⊂C0, and BAND-WIN = "gzippy is Pareto-best among all
> points meeting the target." The partition table is kept below for intuition only; the
> executable semantics are D4's. (The original C0 band `r ≥ r(gzip-1)` had an inverted
> inequality — smaller `r` is better — caught by the review as bug #7.)

Nominal levels lie across tools: `gzip -6`, `igzip -2`, `libdeflate -9`, `zlib-ng -6`,
`pigz -6`, `bgzip -6` all denote "level 6" but land at very different ratios. Comparing
"gzippy -6 vs pigz -6" is a category error. Instead define **four ratio-anchored
classes**, and assign every `(tool, level)` point to a class by its MEASURED ratio on a
frozen reference corpus (silesia decompressed), not by its printed number:

| class | ratio band (anchor: measured silesia ratio r = out/in) | anchored by | gzippy point(s) |
|---|---|---|---|
| **C0 FASTEST** | loosest ratio the field offers — `r ≥ r(gzip -1)` band | gzip -1 as the loose anchor | T1 L1 (libdeflate/ISA-L), T>1 L1–2 |
| **C1 DEFAULT** | the "gzip -6" band — `r(gzip -9) < r ≤ r(gzip -1)`, centered on gzip -6 | gzip -6 | T1 L5–6 (flate2/zlib-ng), T>1 L5–6 |
| **C2 MAX-DEFLATE** | tightest ratio achievable by a *non-exhaustive* DEFLATE — `r(zopfli) < r ≤ r(gzip -9)` | gzip -9 / libdeflate -12 | T1 L9 (flate2), T>1 L9 (pipelined), T>1 L10–12 (libdeflate ultra) |
| **C3 ZOPFLI-CORNER** | exhaustive-search corner — `r ≤ r(zopfli)` | zopfli one-point | L11 (gzippy zopfli, single-member, bit-identical to C zopfli) |

**Assignment procedure (deterministic, per corpus).** Before any wall timing:
(1) run every `(tool, level)` once, record `bytes`; (2) compute `r = bytes/in_bytes`;
(3) the class boundaries are the measured `r` of the four anchors on THAT corpus
(gzip-1, gzip-6, gzip-9, zopfli); (4) each point falls into the band its `r` lands in.
The bands are **corpus-local** (a corpus that compresses hard shifts every anchor
together, which is correct — a class is "the ratio you'd accept for this data," not an
absolute number). gzippy's point in a class = its fastest level whose `r` lands in that
band; if gzippy has multiple levels in a band, keep the Pareto-best (smallest wall among
those tied-or-smaller in bytes).

Why anchor on **gzip** for C0–C2 and **zopfli** for C3: gzip is the lingua-franca
reference every user's mental model of "-1/-6/-9" is calibrated against, and zopfli is
the unambiguous exhaustive-search corner (pigz -11 == zopfli, gzippy L11 == zopfli). The
anchors are TOOLS THAT DEFINE THE BAND SEMANTICALLY, not gzippy's own points (never let
the subject define its own grading curve).

**Rejected alternative (recorded so we don't relitigate):** classing by nominal level
number. Rejected because it pairs non-comparable ratios and would make "gzippy -6 beats
libdeflate -6 on speed" a WIN while gzippy -6 ships a bigger file than libdeflate -6 —
the exact ratio-regression-as-speed-win failure mode this whole section exists to kill.

### 1.7 Aggregate view — the full Pareto frontier plot (per corpus)

Per `(box × corpus × T)` the render emits the **complete point cloud**: every
`(tool, level)` as a `(wall, bytes)` point, plus the computed 2-D lower-left Pareto hull
of the field. "Dominance" at the corpus level = **gzippy's own frontier (its points
across all its levels) sits on-or-above the field hull everywhere** — i.e. for every
segment of the field hull there is a gzippy point at-or-better. This is the plot a human
reads to see "is gzippy's curve the outer envelope." The render additionally lists, for
each field-hull vertex NOT owned by gzippy, the gzippy point nearest it and the axis +
magnitude of the gap (the compression loss-list).

### 1.8 What the scoreboard artifact must gain (minimal, additive)

Per arm, add to the recorded evidence: `output_bytes` (exact, from the correctness
run), `input_bytes` (corpus raw size, already pinned). Per cell, add: `level_class`,
`ratio = output_bytes/input_bytes`, and the pairwise-verdict table vs each rival. The
existing wall fields, sha oracle, A/A, TOST, refusal semantics are UNCHANGED — the bytes
axis is a strictly additional required-evidence field (`output_bytes` missing ⇒ the cell
is `REFUSED`, same funnel as a missing wall). No new statistics primitive is needed on
the bytes axis (integer compare); the wall axis reuses `optgate::sign_test_two_sided` +
TOST exactly as decompression does.

---

## 2. ORACLE (correctness is the veto; a fast wrong file is a loss)

Compression's oracle is a **round-trip**: gzippy's output must decompress to the exact
input, validated by TWO INDEPENDENT decompressors so a bug shared with one library can't
hide.

### 2.1 Round-trip, two independent decompressors

For every gzippy compressed output `C` of input `I`:
1. `gzip -dc C | cmp - I` — GNU gzip/zlib lineage.
2. `libdeflate-gunzip -c C | cmp - I` — libdeflate lineage (independent codebase).
Both must reproduce `I` byte-for-byte (`sha256(decoded) == corpus.decompressed_sha256`,
which is already the pinned corpus sha — the input plaintext). A mismatch on EITHER ⇒
cell `VOID{reason:"roundtrip-fail:<tool>"}`, never a wall/verdict. (This mirrors the
decompression scoreboard's "sha == corpus.decompressed_sha256" gate, run in the
opposite direction.)

### 2.2 RFC-1952 structural validity

Independently of round-trip, assert the output parses as RFC-1952 gzip: magic `1f 8b`,
CM=8, header flags consistent, and the **trailer ISIZE == (input length mod 2^32)** and
**trailer CRC32 == crc32(input)**. This catches a decoder that is lenient about a
malformed trailer that gzippy emits (a round-trip alone can pass on a lenient decoder).
Compute CRC32/ISIZE on the Mac from the pinned input; compare against the bytes gzippy
wrote. Any mismatch ⇒ `VOID{reason:"rfc1952:<field>"}`.

### 2.3 GZ-subfield / multi-block outputs decode EVERYWHERE

gzippy's parallel path emits a "GZ" FEXTRA subfield (`GZ_SUBFIELD_ID`,
`parallel.rs:25–27`) carrying per-block sizes, and the pipelined/parallel paths emit
multi-member or subfield-bearing gzip. The code CLAIMS this is "valid gzip readable by
any decompressor … they ignore unknown subfields" (`parallel.rs:344–345,382`).

> **This claim is CODE-ASSERTED, not measured — it is a `HYPOTHESIS (unvalidated)`
> until the oracle proves it.** The oracle §2.1 tests it directly and mandatorily:
> plain `gzip -dc` (which has zero knowledge of the "GZ" subfield) must round-trip
> every parallel/pipelined/GZ-subfield output. If any tool that a real user would use
> (`gunzip`, `zcat`, `zlib`, `libdeflate`) fails on a GZ-subfield file, that is a P0
> correctness LOSS regardless of wall — the whole "produces standard gzip" premise
> fails. The oracle set therefore includes `gunzip -c`, `zcat`, and Python `gzip`
> module as *additional* decode witnesses for GZ-subfield / multi-member / pipelined
> outputs specifically (they are the outputs where "everywhere-decodable" is at risk;
> single-member ISA-L/flate2/zopfli outputs need only the two §2.1 witnesses).

### 2.4 Correctness/timing separation (inherited)

Timing arms write `stdout → /dev/null` (both arms same sink — Gate 0d); a SEPARATE
untimed correctness run captures the output for the round-trip + RFC-1952 + byte-count
gates. `output_bytes` is measured in the correctness run, never inside the timed region.
This is verbatim the scoreboard's "timing/correctness separated" discipline (design
disposition #9), reused unchanged.

### 2.5 Determinism policy — OPEN QUESTION + recommended default

**OPEN QUESTION:** must gzippy's compressed output be bit-identical for a fixed
`(input, level, T)`, even on the parallel/pipelined multi-thread paths?

**Recommended default: YES — deterministic output per `(input, level, T)`.** Rationale:
(a) the bytes axis in §1.2 is declared EXACT; that is only true if `output_bytes` (and
ideally the bytes themselves) do not wobble with thread scheduling. (b) **pigz is
deterministic** for a fixed `-p` and is the field's credibility bar — a nondeterministic
gzippy would be strictly less trustworthy than the tool it is trying to beat. (c) A
deterministic build is testable (an exact-output regression test), a nondeterministic
one is not.

Note the output legitimately **varies with T** (more threads at L1–2 → smaller blocks →
different Huffman trees → different bytes; `get_optimal_block_size` is a function of
`num_threads`). That is expected and fine — the ratio axis is defined *per cell*, and T
is a cell coordinate. The determinism requirement is only *within* a fixed
`(input, level, T)`.

**Verification the oracle owes (pre-registered):** run the same `(input, level, T)`
compression K≥3 times; assert `sha256(output)` identical across runs. If a
measured win later requires a nondeterministic scheme (e.g. work-stealing that reorders
block emission), that is a deliberate policy change requiring an R3 user decision and a
downgrade of the bytes axis to "exact-per-run, median-reported" — do NOT let it happen
silently. Recommended stance shipped as default: **deterministic; a determinism-breaking
optimization must clear a gated win AND a user sign-off.**

---

## 3. TOOL FIELD + staging plan

Per host. Boxes are named only (no access tonight); staging is what day-1 setup runs.
Everything about thread capability that I cannot verify from the box is marked
**UNVERIFIED** and carries the exact probe that resolves it.

### 3.1 The compressor field

| tool | role | levels | multi-thread? | deterministic? | notes |
|---|---|---|---|---|---|
| **gzippy** (subject) | subject | 0–12 (11=zopfli via `--ultra`/flags) | yes (`-p N`) | recommended YES (§2.5) | the tool under test |
| **gzip** | C0–C2 anchor + rival | 1–9 | no | yes | defines the C0/C1/C2 band anchors |
| **pigz** | primary parallel rival | 0–11 (11=zopfli) | **yes** `-p N` | **yes** (per fixed -p) | the parallel bar; deterministic bar for §2.5 |
| **igzip** (ISA-L) | fast-class rival | 0–3 | `-T N` **UNVERIFIED** | UNVERIFIED | the ISA-L path gzippy's own L0–3 wraps; head-to-head at the fast corner |
| **libdeflate-gzip** | ratio-at-speed rival | 0–12 | **no** (single-thread only) | yes | strong C2 point at T1; no parallel arm |
| **zlib-ng** (`minigzip-ng`) | baseline codec rival | 0–9 | no | yes | gzippy's own C1 backend (flate2/zlib-ng); head-to-head sanity |
| **zopfli** | C3 anchor | single point (`--i N` iters) | no | yes | defines the C3 band; gzippy L11 should be bit-identical |
| **bgzip** (htslib) | parallel-block rival | 0–9 (libdeflate-backed) | `-@ N` **UNVERIFIED** | UNVERIFIED | closest analogue to gzippy's GZ-subfield parallel-decodable format |

**Honest capability notes (mark and probe, do not assume):**
- **igzip `-T`**: newer isa-l `igzip` CLI exposes `-T <n>` threads — **UNVERIFIED on our
  staged binary.** Probe: `igzip --help 2>&1 | grep -i thread`; if absent, igzip is a
  T1-only rival (compare it only in T1 cells, like libdeflate/zlib-ng).
- **bgzip `-@`**: htslib `bgzip` exposes `-@ <n>` threads and `-l <level>` — **version
  and flag UNVERIFIED on our staged binary.** Probe: `bgzip --help 2>&1 | grep -E
  '@|level'`. bgzip output is BGZF (its own parallel-decodable format), directly
  comparable to gzippy's GZ-subfield trade.
- **pigz `-p`**: verified-by-reputation (documented, widely relied on) but still
  **probe on the staged binary**: `pigz --version` + a `-p 4` smoke. pigz determinism
  for fixed `-p` is documented; the oracle's §2.5 determinism test covers it anyway.
- **libdeflate-gzip / zlib-ng / gzip / zopfli**: single-thread — appear only in T1
  cells (and in every T cell as the T1-invariant rival reference for the hull, since a
  user could always fall back to them).

### 3.2 Per-host staging

- **M1 local (Apple M1 Pro)** — `fulcrum-mac` LocalRunner smoke host. **ISA-L is
  unavailable on arm64**, so gzippy's C0 path there is libdeflate, NOT ISA-L
  (`isal_compress::is_available()==false`). Field to stage via Homebrew/source:
  `gzip` (system), `pigz`, `libdeflate` tools, `zopfli`, `bgzip` (htslib), `zlib-ng`
  minigzip-ng, `igzip` (isa-l — **may not build/run usefully on arm64; mark
  UNVERIFIED**, likely absent → M1 field drops igzip). Threads `[1,4,8]` (8 P-cores).
  Role: smoke the whole 2-D pipeline + the arm64-specific C0 story.
- **solvency (AMD EPYC 7282, Zen2)** — running the user's llama; **do NOT pause**;
  measure LOAD-IMMUNE (quiesce spec opt-in per scoreboard, or the abmeasure
  contention-invariant path). x86_64 ⇒ ISA-L + AVX2 available (gzippy C0 = ISA-L).
  Full field incl. igzip. Threads `[1,4,8,16]`.
- **neurotic (Intel i7-13700, freezable)** — 7 independent P-cores. Full field incl.
  igzip. Threads `[1,4,7]`. Freeze/quiet gates available (`freeze_state:frozen`).

Version pins are recorded per staged binary at run time via the scoreboard's box-local
`sha256sum <bin>` + `<tool> --version` capture (provenance block). NO version is
asserted here from memory — the staging step DERIVES them and the artifact refuses a
cell whose comparator hash/version could not be measured (scoreboard disposition #5).

---

## 4. CORPUS

Reuse the decompression set (feed the **decompressed/plaintext** forms as compression
INPUT), plus three compression-specific inputs that exercise routing branches
decompression never touches.

### 4.1 Reused (plaintext inputs = the decompressed decomp corpora)

| id | what | why (compression-relevant) |
|---|---|---|
| `silesia` | Silesia mix | the balanced ratio reference; **defines the C0–C2 band anchors** (§1.6). Primary. |
| `monorepo` | source-tree tar | highly-redundant text; separates dictionary-continuation (pipelined) from independent-block (parallel) ratio cost sharply. |
| `nasa` | numeric/binary | low-redundancy binary; stresses match-finder + the fast classes. |
| `squishyreal` | real-world blobs | broad realistic spread. |
| `squishycombined` | combined corner | corner mix; retained for continuity with the decomp field. |

### 4.2 Compression-specific additions (justify each)

| id | input | what it tests | why it MUST exist |
|---|---|---|---|
| `pregz` | an already-gzipped file (feed a `.gz` as INPUT) | **recompression / store-detection**: incompressible-to-gzip input (already entropy-coded). Exercises the ratio-probe branch (`mod.rs:79–91`, `(actual/len) >= 0.10 → libdeflate` vs flate2) and whether gzippy emits near-stored output instead of wasting cycles. | A real user pipeline double-gzips constantly (`tar.gz` re-packed). If gzippy inflates size or burns time vs `gzip` on already-compressed data, that is a visible loss no plaintext corpus reveals. |
| `zeros` | a large all-zero / sparse file | **max-ratio + RLE corner**: hits the "highly compressible" flate2 branch and the L1 RLE mapping (`adjust_compression_level`, L1→L2). Ratio is enormous; the interesting axis is wall + whether block-reset (parallel path) wastes ratio even here. | The extreme-compressible end of the ratio axis; guarantees the C0 band's loose anchor is exercised and surfaces any per-block-tree overhead that only shows when payload→0. |
| `rand` | a PRNG-incompressible file (`/dev/urandom` snapshot, pinned) | **worst-case ratio ≈ 1.0 / store-detection**: forces the `< 0.10`-compressible → flate2 vs `>= 0.10` → libdeflate decision to the incompressible side; every tool should fall back to ~stored. | The other extreme of the ratio axis; tests that gzippy never emits a file LARGER than input (a stored-block correctness/quality bar) and that store-detection wall is competitive. |

All three additions are pinned (`sha256` of the exact bytes) so the bytes axis stays
exact and reproducible across boxes. `pregz` is generated once (`gzip -6 silesia >
pregz.in`) and pinned so every box compresses byte-identical input.

---

## 5. DOMINANCE SPECS (executable finish lines)

Two specs, both consumed by a proposed `fulcrum scoreboard diff --dominance <spec>`
mode: instead of diffing two artifacts, `diff --dominance` loads ONE artifact (a
`scoreboard run` output) and checks each declared finish-line cell against it,
emitting `PASS`/`FAIL`/`MISSING` per cell and a nonzero exit if any `FAIL`/`MISSING`
(CI-gate shape, same as `diff`'s regression exit).

### 5.1 Minimal schema extension (do NOT implement — propose only)

```
{
  "dominance_version": 1,
  "kind": "decompression" | "compression",
  "subject": "<subject label, must match artifact subject>",
  "protocol": "<must equal the artifact protocol version>",
  "defaults": { "require": "<default requirement>", "vs": ["*"] },
  "cells": [
     { "box": "<id>", "corpus": "<id>", "threads": <int>,
       "level_class": "<C0|C1|C2|C3>"        // compression only; omitted for decompression
       "require": "WIN" | "TIE-OR-BETTER" | "HULL-MEMBER" | "BAND-WIN",
       "vs": ["*"] | ["igzip","pigz",...]    // which comparators must be beaten/tied
     }
  ],
  "pass_condition": "all_cells_satisfied"
}
```

Requirement vocabulary (evaluated against the artifact's per-cell verdicts):
- `WIN` — cell verdict must be `WIN` vs every tool in `vs`.
- `TIE-OR-BETTER` — cell verdict `WIN` or `TIE` (never LOSS/CONTESTED/VOID).
- `HULL-MEMBER` — (compression) gzippy point not strictly dominated by any rival in the
  corpus; i.e. cell verdict ∈ `{WIN, TIE, CONTESTED-but-on-hull}` and NOT `LOSS`.
- `BAND-WIN` — (compression) cell verdict `WIN` in the named `level_class`.

`diff --dominance` needs ONLY read fields the artifact already records under this
design (`§1.8` verdicts + `output_bytes`); it computes no new statistics. A cell present
in the spec but absent in the artifact ⇒ `MISSING` (fail) — the finish line refuses to
pass on un-measured cells, mirroring the scoreboard's refusal ethos.

### 5.2 The two spec files (in scratchpad)

- `dominance-spec-decomp.json` — kind=decompression, enumerates every
  `(box × corpus × T)` from the baseline field (solvency T[1,4,8,16], neurotic
  T[1,4,7], m1 T[1,4,8]) × corpora {silesia,monorepo,nasa,squishyreal,squishycombined},
  each `require: TIE-OR-BETTER, vs: ["*"]` (the whole field: igzip, libdeflate, zlibng,
  pigz, rapidgzip). This is the decompression finish line the current campaign is
  closing — expressed as a checkable artifact so "we're done" becomes a tool verdict,
  not a claim.
- `dominance-spec-comp.json` — kind=compression, enumerates `(box × corpus × T ×
  level_class)`; every cell `require: HULL-MEMBER` (the mandatory floor) with a parallel
  `stretch` block listing the same cells at `require: BAND-WIN` (the aspirational
  finish, checked but non-blocking via `pass_condition: hull_all + bandwin_report`).

---

## 6. DAY-1 RUNBOOK (ordered)

Exact steps to run the moment the decompression gap closes and compression opens. No
tooling build — all specs already exist.

1. **Stage the field** on all three hosts (§3.2). For each staged binary run the
   capability probes (§3.1 igzip `-T`, bgzip `-@`, pigz `-p` smoke) and record
   `--version` + `sha256sum`. Resolve every UNVERIFIED flag → concrete `threads_max`
   in the run spec. Generate + pin the three compression-specific corpora (§4.2).
2. **Determinism pre-flight** (oracle §2.5): for one cell per host, compress the same
   `(input, level, T)` K=3× and assert identical `sha256(output)`. A failure here BLOCKS
   the baseline (the bytes axis is not exact until this passes) → this is the first
   thing that can send us to an R3 determinism decision.
3. **Baseline run, all three hosts** — `fulcrum scoreboard run --spec
   compression-spec-draft.json` (interleaved best-of-N, /dev/null timed sink,
   correctness arm captures `output_bytes` + round-trips both decompressors + RFC-1952).
   solvency uses the quiesce opt-in (no-orphan lives in the tool); neurotic uses
   freeze/quiet; M1 is the LocalRunner smoke that proves the 2-D loop end-to-end.
4. **Generate the loss-map** — `fulcrum scoreboard render <artifact>`: per
   `(box × corpus × T)` the full (wall, bytes) point cloud + field hull + the
   compression LOSS-LIST (every cell where gzippy is off-hull or CONTESTED-behind,
   sorted by the worse-axis deficit). This REPLACES any hand-written compression
   loss-map (memory rule: loss-map is GENERATED).
5. **Front classification playbook** — for each LOSS/off-hull cell, apply the
   inherited discipline: is the deficit on the WALL axis (→ a speed lever, use `fulcrum
   perturb` removal-oracle to bound it) or the BYTES axis (→ a ratio lever: a
   structural ratio gap is deterministic, no significance test — measure the byte
   delta and locate the cause: block-reset vs dictionary-continuation, tree-per-block
   overhead, etc.)? Ratio losses and wall losses have DIFFERENT playbooks; the render
   tags each.
6. **First lever** — pick the worst HULL-MEMBER violation (a true LOSS, gzippy
   strictly dominated) first — that is a cell where a user is unambiguously better off
   with a rival, the highest-value gap. Pre-register its hypothesis + falsifier before
   touching code (bias rule). Only after the mandatory HULL floor is met everywhere do
   BAND-WIN (stretch) cells become the work queue, VoI-ordered.

### 6.1 Top-3 PREDICTED compression fronts (each a HYPOTHESIS + its test)

These are read off gzippy's compress source; they are `HYPOTHESIS (unvalidated)` and
their ONLY licensed next action is the measurement named. None is a finding.

1. **HYPOTHESIS (unvalidated): parallel-path ratio loss vs pigz (BYTES axis, C0/C1,
   T>1).** `ParallelGzEncoder` resets the DEFLATE dictionary at every independent block
   boundary (`parallel.rs:15`, per-block gzip members), whereas pigz continues a 32 KiB
   dictionary across blocks. Predicts gzippy T>1 L1–6 emits a *larger* file than pigz at
   the same speed class — a CONTESTED-or-LOSS on the bytes axis that grows with thread
   count (smaller blocks → more resets). *Test:* compression scoreboard cell
   `(silesia × T∈{4,8,16} × C0,C1)`, gzippy `ParallelGzEncoder` vs `pigz -p T`; compare
   EXACT `output_bytes` (deterministic, no significance needed) and wall. If bytes_g >
   bytes_pigz while wall ties → CONTESTED front; magnitude = the ratio deficit. Confirm
   the mechanism by A/B on `monorepo` (high-redundancy amplifies dictionary value → the
   gap should widen).
2. **HYPOTHESIS (unvalidated): ISA-L C0 head-to-head — gzippy-wrapping-ISA-L vs igzip
   itself (WALL axis, T1 L0–3, x86 only).** gzippy's T1 L0–3 path calls ISA-L
   (`isal_compress::compress_gzip_stream_direct`) behind its own framing/probe; igzip is
   ISA-L raw. Predicts gzippy adds framing/probe overhead over bare igzip at the fast
   corner → a WALL-axis TIE-or-slight-LOSS at equal bytes. *Test:* cell `(silesia,nasa ×
   T1 × C0)` gzippy L1–3 vs igzip L1–3 on solvency/neurotic (x86); paired wall sign-test
   + TOST; bytes should be BYTE-EQUAL or near (both ISA-L) so this is a clean 1-D wall
   question. On M1 (no ISA-L) the same cell tests gzippy-libdeflate vs a libdeflate rival
   instead — a *different* front, recorded separately (arch dispatch).
3. **HYPOTHESIS (unvalidated): max-ratio corner — gzippy L9/L10–12/L11 vs zopfli &
   libdeflate-12 (BYTES axis, C2/C3).** In C2, gzippy T>1 L6–9 uses the pipelined
   zlib-ng path (pigz-ratio) and L10–12 uses libdeflate ultra; the question is whether
   either matches libdeflate-12's ratio at competitive wall, and whether L11 is truly
   bit-identical to C zopfli (a correctness+ratio claim in `zopfli.rs:1–14`). Predicts a
   possible bytes gap where libdeflate-12 (single-thread) out-compresses gzippy's
   pipelined C2 point. *Test:* cell `(silesia,monorepo × T1 and Tmax × C2,C3)`; exact
   `output_bytes` gzippy-L9/L12/L11 vs libdeflate-12, zopfli, pigz-11; on C3 assert
   gzippy L11 output `sha256 == ` C-zopfli output (the bit-identity claim) — a
   deterministic pass/fail, not a race.

---

## 7. ADVERSARIAL REVIEW (round 1, cursor-agent gpt-5.5-high)

Focused on the two poison points: the 2-D Pareto verdict semantics (§1.3–1.5) and the
level-class mapping (§1.6). Dispositions recorded below.

The reviewer raised 11 concrete flaws; **all 11 ADOPTED**. Two are structural
(they AMEND §1.3–1.6 above — the amendments below govern where they conflict); the
rest harden edges. A real direction bug (#7) was caught.

**D1 — WALL-VOID must propagate as INDETERMINATE, never collapse to MATCH/CONTESTED
(amends §1.2–1.4).** Add a pairwise `INDETERMINATE` and a cell `INDETERMINATE`. Any
required comparison with `WALL-VOID` blocks `WIN`/`TIE`/`LOSS` UNLESS the bytes axis
alone settles it irreversibly (e.g. gzippy strictly LARGER and no faster-possible →
still not dominated only if wall could rescue it; be conservative → INDETERMINATE and
re-measure). INDETERMINATE never counts toward campaign completion — it is a
"re-measure" state, distinct from a certified TIE.

**D2 — WALL-TIE means TOST-EQUIVALENT under the pre-declared margin, NOT merely
"sign-test not significant" (amends §1.2).** An underpowered/insignificant wall test is
`WALL-VOID`, not `WALL-TIE`. Only a passed TOST (paired log-ratio CI inside
`criteria.tie_margin_pct` AND `aa_spread ≤ aa_spread_cap_pct`) yields `WALL-TIE`. This
is already the scoreboard's TOST rule; §1.2 is tightened to forbid "not significant ⇒
tie." This is the exact "signal-moves-wall-doesn't false-TIE" hazard from memory.

**D3 — FRONTIER semantics replace "one gzippy point vs one rival point" (amends
§1.3–1.6, STRUCTURAL).** Per class, each tool contributes its full in-band Pareto
FRONTIER (multiple points), not a single "fastest level." Before aggregating: (a)
compute the field's admissible frontier = union of all tools' points, drop every
strictly-dominated rival point; (b) gzippy WINS a rival point only by
dominating/matching a NON-DOMINATED (on-frontier) rival point; (c) a
strictly-dominated (off-hull) rival can NEVER by itself force gzippy's cell to
CONTESTED or LOSS — it is filtered out first (fixes #3 inflated-WIN and #4
dominated-rival-poisons-CONTESTED). Cell verdict is gzippy's frontier vs the field's
admissible frontier, worst-case over the ADMISSIBLE rivals only.

**D4 — Level-classes are NESTED QUALITY-TARGET CONSTRAINTS, not a partition (amends
§1.6, STRUCTURAL; fixes the #7 direction bug).** Recall `r = out/in`, so SMALLER r =
better compression, and anchors sort `r(gzip-1) ≥ r(gzip-6) ≥ r(gzip-9) ≥ r(zopfli)`.
Redefine each class as a ratio **target (a quality floor)**:
- C0 target = `r ≤ r(gzip-1)` (at least gzip-1 quality),
- C1 target = `r ≤ r(gzip-6)`, C2 target = `r ≤ r(gzip-9)`, C3 target = `r ≤ r(zopfli)`.
Targets are NESTED (C3 ⊂ C2 ⊂ C1 ⊂ C0). **BAND-WIN(class) = among ALL (tool,level)
points meeting that class's target, a gzippy point is Pareto-best (fastest at-or-below
the target with no rival simultaneously smaller AND faster).** This kills the original
§1.6 partition (which had the inverted `r ≥ r(gzip-1)` C0 band that admitted
WORSE-than-gzip-1 outputs — the #7 bug) and the "band too wide hides a ratio gap" (#6):
a constraint target compares only points that clear the SAME quality bar, and the exact
bytes axis still separates them within it. gzippy's representative point in a class =
its Pareto-best point meeting the target (it may have several; keep the frontier per D3).

**D5 — Validate anchors per corpus; handle collapse/inversion (amends §1.6).** Before
classing, sort the four anchor ratios; assert monotonic non-increasing. On a
pathological corpus where anchors invert or coincide (e.g. `r(zopfli) ≥ r(gzip-9)`),
mark the affected class boundary `INVALID` and MERGE adjacent targets, recording the
merge — never silently mis-class. Half-open handling is moot under the D4 constraint
framing (targets are `≤`, nested), which is another reason to prefer it.

**D6 — No-gzippy-candidate is INELIGIBLE (a failed claim), not a silent skip (amends
§5, comp spec).** If gzippy could meet a class target but its measured points don't, the
cell FAILS (not "skip"). A true N/A exists only when NO tool can meet the target on that
corpus (target unachievable) — recorded with that reason. gzippy structurally always has
a C3 point (L11==zopfli) so C3 is never gzippy-skippable. Comp dominance spec updated:
`on_skipped_class` narrowed to "no tool meets target," and a gzippy-absent-but-achievable
class is `FAIL`.

**D7 — HULL-MEMBER uses the SAME admissible universe, and a global all-level hull is
reported separately (amends §1.5, §1.7).** HULL-MEMBER (the floor) is evaluated on the
GLOBAL field frontier (all tools, all levels) — "gzippy not strictly dominated
anywhere." The per-cell WIN/LOSS/CONTESTED is a within-target-class statement. These are
DIFFERENT questions and both are reported and labelled as such; they cannot "contradict"
because they answer different scopes. The render prints both the per-class cell verdicts
and the one global 2-D hull per (box×corpus×T) — the §1.7 plot IS that global hull.

**D8 — CANONICALIZE header-metadata bytes; COUNT structural-framing bytes (amends §1.2,
§2; the sharpest practical fix).** The exact bytes axis is only meaningful if
metadata-only deltas (mtime, filename, OS byte, FCOMMENT) don't masquerade as ratio.
Policy: invoke EVERY tool with mtime/name suppressed (`gzip -n` and equivalents; gzippy
with no header info) so header-metadata bytes are canonical and constant across tools.
BUT structural framing bytes are a REAL cost the user pays and MUST count: gzippy's "GZ"
FEXTRA subfield and multi-member/pipelined headers add genuine size in exchange for
parallel-decodability — that is exactly the ratio-vs-speed trade the 2-D verdict exists
to price, so those bytes stay IN the count. Decision: **count full `.gz` output bytes
with header METADATA canonicalized (mtime=0, no name/comment) across all tools;
structural framing bytes counted as-is.** Recorded as a required run-config
(`canonicalize_metadata: true`) in the comp spec; the determinism preflight (§2.5) also
guards it (canonical output is a precondition of bit-reproducibility).

**Net effect on the deliverables:** §1.6's partition is SUPERSEDED by D4's nested
targets; §1.3–1.4's single-point pairing is SUPERSEDED by D3's frontier semantics; §1.2
gains INDETERMINATE (D1) and the tightened WALL-TIE (D2); §2 gains the canonicalization
policy (D8). The JSON specs were patched: `compression-spec-draft.json` gains
`canonicalize_metadata` + the `level_classes` note now reads "nested target r ≤
anchor"; `dominance-spec-comp.json` `verdict_evaluation` now uses frontier + nested-
target language and `on_skipped_class` is narrowed per D6. No change was needed to the
decompression spec (1-D wall only). None of the 11 required an architectural redesign of
the scoreboard extension — they refine the verdict function, which is the whole point of
running the review before day 1.
